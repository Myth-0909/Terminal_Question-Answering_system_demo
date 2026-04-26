"""Generate mock Chinese profile data for vector indexing."""

from __future__ import annotations

import json
import random
from typing import Any

from faker import Faker

fake = Faker("zh_CN")

GENDERS = ["男", "女"]
EDUCATIONS = ["高中", "大专", "本科", "硕士", "博士"]
PERSONALITIES = ["外向开朗", "沉稳内敛", "温柔细腻", "幽默风趣", "理性务实", "积极乐观"]
RELATIONSHIP_STATUS = ["单身", "恋爱中", "已婚", "保密"]
OCCUPATIONS = [
    "产品经理",
    "后端工程师",
    "前端工程师",
    "UI设计师",
    "数据分析师",
    "教师",
    "医生",
    "运营专员",
    "销售顾问",
    "短视频创作者",
]
HOBBIES_POOL = [
    "跑步",
    "健身",
    "摄影",
    "手绘",
    "阅读",
    "旅行",
    "烘焙",
    "露营",
    "羽毛球",
    "追剧",
    "桌游",
    "吉他",
    "电影",
    "编程",
    "书法",
    "游泳",
]
FOODS = ["火锅", "烧烤", "麻辣烫", "寿司", "粤菜", "川菜", "牛排", "轻食", "面食"]
MUSICS = ["流行", "民谣", "摇滚", "古典", "说唱", "电子", "轻音乐", "国风"]


def generate_mock_profiles(count: int = 1000, seed: int = 42) -> list[dict[str, Any]]:
    random.seed(seed)
    Faker.seed(seed)
    profiles: list[dict[str, Any]] = []

    for _ in range(count):
        city = fake.city_name()
        hobbies = random.sample(HOBBIES_POOL, k=random.randint(2, 5))
        profile = {
            "name": fake.name(),
            "age": random.randint(18, 55),
            "gender": random.choice(GENDERS),
            "city": city,
            "occupation": random.choice(OCCUPATIONS),
            "education": random.choice(EDUCATIONS),
            "hobbies": hobbies,
            "personality": random.choice(PERSONALITIES),
            "favorite_food": random.choice(FOODS),
            "favorite_music": random.choice(MUSICS),
            "relationship_status": random.choice(RELATIONSHIP_STATUS),
            "bio": fake.sentence(nb_words=18),
        }
        profiles.append(profile)
    return profiles


def profile_to_text(profile: dict[str, Any]) -> str:
    return (
        f"姓名：{profile['name']}；年龄：{profile['age']}；性别：{profile['gender']}；"
        f"城市：{profile['city']}；职业：{profile['occupation']}；学历：{profile['education']}；"
        f"爱好：{', '.join(profile['hobbies'])}；性格：{profile['personality']}；"
        f"喜欢的食物：{profile['favorite_food']}；喜欢的音乐：{profile['favorite_music']}；"
        f"感情状态：{profile['relationship_status']}；个人简介：{profile['bio']}"
    )


def dump_profiles_to_json(path: str, profiles: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
