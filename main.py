import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from fastapi.middleware.cors import CORSMiddleware

# Set these environment variables before starting the server.
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

driver = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global driver
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    driver.verify_connectivity()
    try:
        yield
    finally:
        driver.close()


app = FastAPI(
    title="Fitness Knowledge Graph API",
    description=(
        "以 Neo4j 健身知識圖譜提供動作搜尋、替代動作、"
        "肌肉反向查詢、動作比較、關係探索與簡易課表產生。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_query(query: str, params: Optional[dict] = None):
    """Run a Cypher query and return records as dictionaries."""
    with driver.session(database="neo4j") as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]


EXERCISE_DETAILS_QUERY = """
MATCH (e:Exercise {name: $name})
OPTIONAL MATCH (e)-[r:TARGETS]->(m:Muscle)
WITH e, collect(DISTINCT CASE
    WHEN m IS NULL THEN null
    ELSE {name: m.name, bodyPart: m.bodyPart, role: r.role}
END) AS targets
OPTIONAL MATCH (e)-[:USES]->(eq:Equipment)
RETURN e.name AS name,
       e.difficulty AS difficulty,
       targets,
       collect(DISTINCT eq.name) AS equipment
"""


@app.get("/", tags=["Health"])
def root():
    return {"message": "Fitness API is running", "docs": "/docs"}


# 1. Multi-condition exercise search
@app.get("/exercises", tags=["Exercises"])
def search_exercises(
    body_part: Optional[str] = Query(default=None, description="Muscle.bodyPart 的實際值"),
    role: Optional[str] = Query(default=None, description="例如：主動肌、協同肌"),
    equipment: Optional[str] = Query(default=None, description="Equipment.name 的實際值"),
    difficulty: Optional[str] = Query(default=None, description="Exercise.difficulty 的實際值"),
):
    query = """
    MATCH (e:Exercise)
    WHERE
        ($difficulty IS NULL OR e.difficulty = $difficulty)
        AND (
            $body_part IS NULL
            OR EXISTS {
                MATCH (e)-[r:TARGETS]->(m:Muscle)
                WHERE m.bodyPart = $body_part
                  AND ($role IS NULL OR r.role = $role)
            }
        )
        AND (
            $role IS NULL
            OR $body_part IS NOT NULL
            OR EXISTS {
                MATCH (e)-[r:TARGETS]->(:Muscle)
                WHERE r.role = $role
            }
        )
        AND (
            $equipment IS NULL
            OR EXISTS {
                MATCH (e)-[:USES]->(eq:Equipment)
                WHERE eq.name = $equipment
            }
        )
    OPTIONAL MATCH (e)-[r:TARGETS]->(m:Muscle)
    WITH e, collect(DISTINCT CASE
        WHEN m IS NULL THEN null
        ELSE {name: m.name, bodyPart: m.bodyPart, role: r.role}
    END) AS targets
    OPTIONAL MATCH (e)-[:USES]->(eq:Equipment)
    RETURN e.name AS name,
           e.difficulty AS difficulty,
           targets,
           collect(DISTINCT eq.name) AS equipment
    ORDER BY name
    """
    params = {
        "body_part": body_part,
        "role": role,
        "equipment": equipment,
        "difficulty": difficulty,
    }
    return {"count": len(rows := run_query(query, params)), "data": rows}


# 4. Compare is declared before /exercises/{name} to avoid route ambiguity.
@app.get("/exercises/compare", tags=["Exercise comparison"])
def compare_exercises(
    name1: str = Query(..., description="第一個動作的完整名稱"),
    name2: str = Query(..., description="第二個動作的完整名稱"),
):
    query = """
    UNWIND [$name1, $name2] AS requestedName
    MATCH (e:Exercise {name: requestedName})
    OPTIONAL MATCH (e)-[r:TARGETS]->(m:Muscle)
    WITH e, collect(DISTINCT CASE
        WHEN m IS NULL THEN null
        ELSE {name: m.name, bodyPart: m.bodyPart, role: r.role}
    END) AS targets
    OPTIONAL MATCH (e)-[:USES]->(eq:Equipment)
    RETURN e.name AS name,
           e.difficulty AS difficulty,
           targets,
           collect(DISTINCT eq.name) AS equipment
    """
    rows = run_query(query, {"name1": name1, "name2": name2})
    by_name = {row["name"]: row for row in rows}
    missing = [name for name in (name1, name2) if name not in by_name]
    if missing:
        raise HTTPException(status_code=404, detail=f"找不到動作：{', '.join(missing)}")

    first, second = by_name[name1], by_name[name2]
    muscles1 = {x["name"] for x in first["targets"] if x and x.get("name")}
    muscles2 = {x["name"] for x in second["targets"] if x and x.get("name")}
    equipment1 = set(first["equipment"])
    equipment2 = set(second["equipment"])

    return {
        "exercise_1": first,
        "exercise_2": second,
        "comparison": {
            "common_muscles": sorted(muscles1 & muscles2),
            "only_in_exercise_1": sorted(muscles1 - muscles2),
            "only_in_exercise_2": sorted(muscles2 - muscles1),
            "common_equipment": sorted(equipment1 & equipment2),
            "only_equipment_for_exercise_1": sorted(equipment1 - equipment2),
            "only_equipment_for_exercise_2": sorted(equipment2 - equipment1),
            "same_difficulty": first["difficulty"] == second["difficulty"],
        },
    }


# 2. Find movements for a target muscle that use available equipment.
@app.get("/exercises/alternatives", tags=["Recommendations"])
def alternative_exercises(
    muscle: str = Query(..., description="目標 Muscle.name"),
    equipment: str = Query(..., description="可用 Equipment.name"),
    exclude: Optional[str] = Query(default=None, description="可選：要排除的動作名稱"),
    role: Optional[str] = Query(default=None, description="可選：主動肌或協同肌"),
):
    query = """
    MATCH (e:Exercise)-[r:TARGETS]->(m:Muscle)
    WHERE m.name = $muscle
      AND ($role IS NULL OR r.role = $role)
      AND ($exclude IS NULL OR e.name <> $exclude)
      AND EXISTS {
          MATCH (e)-[:USES]->(eq:Equipment)
          WHERE eq.name = $equipment
      }
    WITH DISTINCT e
    OPTIONAL MATCH (e)-[r2:TARGETS]->(m2:Muscle)
    WITH e, collect(DISTINCT CASE
        WHEN m2 IS NULL THEN null
        ELSE {name: m2.name, bodyPart: m2.bodyPart, role: r2.role}
    END) AS targets
    OPTIONAL MATCH (e)-[:USES]->(eq2:Equipment)
    RETURN e.name AS name,
           e.difficulty AS difficulty,
           targets,
           collect(DISTINCT eq2.name) AS equipment
    ORDER BY name
    """
    rows = run_query(query, {
        "muscle": muscle,
        "equipment": equipment,
        "exclude": exclude,
        "role": role,
    })
    return {"count": len(rows), "data": rows}


# 3. Reverse lookup: which exercises target a muscle?
@app.get("/muscles/{name}/exercises", tags=["Muscles"])
def exercises_for_muscle(
    name: str,
    role: Optional[str] = Query(default=None, description="可選：主動肌、協同肌"),
):
    query = """
    MATCH (e:Exercise)-[r:TARGETS]->(m:Muscle {name: $name})
    WHERE $role IS NULL OR r.role = $role
    WITH DISTINCT e, r, m
    OPTIONAL MATCH (e)-[:USES]->(eq:Equipment)
    RETURN e.name AS name,
           e.difficulty AS difficulty,
           m.name AS muscle,
           m.bodyPart AS bodyPart,
           r.role AS role,
           collect(DISTINCT eq.name) AS equipment
    ORDER BY name
    """
    rows = run_query(query, {"name": name, "role": role})
    if not rows:
        # Distinguish an unknown muscle from a muscle with no connected exercises.
        exists = run_query(
            "MATCH (m:Muscle {name: $name}) RETURN m.name AS name",
            {"name": name},
        )
        if not exists:
            raise HTTPException(status_code=404, detail=f"找不到肌肉：{name}")
    return {"count": len(rows), "data": rows}


# 5. Explore the graph from an equipment node through exercises to muscles.
@app.get("/graph/equipment/{name}/exercises", tags=["Graph exploration"])
def graph_from_equipment(name: str):
    query = """
    MATCH (eq:Equipment {name: $name})<-[:USES]-(e:Exercise)
    OPTIONAL MATCH (e)-[r:TARGETS]->(m:Muscle)
    RETURN eq.name AS equipment,
           e.name AS exercise,
           e.difficulty AS difficulty,
           collect(DISTINCT CASE
               WHEN m IS NULL THEN null
               ELSE {name: m.name, bodyPart: m.bodyPart, role: r.role}
           END) AS targets
    ORDER BY exercise
    """
    rows = run_query(query, {"name": name})
    if not rows:
        exists = run_query(
            "MATCH (eq:Equipment {name: $name}) RETURN eq.name AS name",
            {"name": name},
        )
        if not exists:
            raise HTTPException(status_code=404, detail=f"找不到器材：{name}")
    return {"count": len(rows), "data": rows}


# 6. Simple rule-based workout-plan generator.
class PlanRequest(BaseModel):
    body_parts: list[str] = Field(
        min_length=1,
        description="訓練部位清單，值需符合 Muscle.bodyPart",
        examples=[["胸部", "肩部"]],
    )
    equipment: list[str] = Field(
        min_length=1,
        description="可用器材名稱清單，值需符合 Equipment.name",
        examples=[["啞鈴", "徒手"]],
    )
    days: int = Field(default=3, ge=1, le=7, description="每週訓練天數")
    exercises_per_day: int = Field(default=4, ge=1, le=12, description="每天安排的動作數")
    difficulty: Optional[str] = Field(default=None, description="可選：難度需符合 Exercise.difficulty")


@app.post("/plans/generate", tags=["Workout plans"])
def generate_plan(request: PlanRequest):
    query = """
    MATCH (e:Exercise)-[:TARGETS]->(m:Muscle)
    WHERE m.bodyPart IN $body_parts
      AND ($difficulty IS NULL OR e.difficulty = $difficulty)
      AND EXISTS {
          MATCH (e)-[:USES]->(eq:Equipment)
          WHERE eq.name IN $equipment
      }
    WITH DISTINCT e
    OPTIONAL MATCH (e)-[r:TARGETS]->(m2:Muscle)
    WITH e, collect(DISTINCT CASE
        WHEN m2 IS NULL THEN null
        ELSE {name: m2.name, bodyPart: m2.bodyPart, role: r.role}
    END) AS targets
    OPTIONAL MATCH (e)-[:USES]->(eq2:Equipment)
    RETURN e.name AS name,
           e.difficulty AS difficulty,
           targets,
           collect(DISTINCT eq2.name) AS equipment
    ORDER BY name
    """
    candidates = run_query(query, {
        "body_parts": request.body_parts,
        "equipment": request.equipment,
        "difficulty": request.difficulty,
    })

    if not candidates:
        return {
            "message": "找不到符合條件的動作。請確認部位、器材與難度名稱是否和資料庫一致。",
            "requested_days": request.days,
            "plans": [],
            "candidate_count": 0,
        }

    # Deterministic first version: distribute available unique exercises across
    # days. Do not silently duplicate exercises if the database has too few.
    plans = []
    for day_index in range(request.days):
        start = day_index * request.exercises_per_day
        end = start + request.exercises_per_day
        plans.append({
            "day": day_index + 1,
            "exercises": candidates[start:end],
        })

    return {
        "message": "這是依資料庫條件產生的簡易課表草案，請依個人能力調整。",
        "requested_days": request.days,
        "exercises_per_day": request.exercises_per_day,
        "candidate_count": len(candidates),
        "plans": plans,
    }


@app.get("/muscles", tags=["Muscles"])
def get_muscles():
    query = """
    MATCH (m:Muscle)
    RETURN DISTINCT m.name AS name, m.bodyPart AS bodyPart
    ORDER BY bodyPart, name
    """
    rows = run_query(query)
    return {"count": len(rows), "data": rows}


@app.get("/equipment", tags=["Equipment"])
def get_equipment():
    query = """
    MATCH (eq:Equipment)
    RETURN DISTINCT eq.name AS name
    ORDER BY name
    """
    rows = run_query(query)
    return {"count": len(rows), "data": rows}


@app.get("/exercises/{name}", tags=["Exercises"])
def get_exercise(name: str):
    rows = run_query(EXERCISE_DETAILS_QUERY, {"name": name})
    if not rows:
        raise HTTPException(status_code=404, detail=f"找不到動作：{name}")
    return rows[0]
