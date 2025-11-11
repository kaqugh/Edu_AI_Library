import csv
import random
import os

# --- Ensure data folder exists ---
os.makedirs(os.path.expanduser('~/Edu_AI_Library/data'), exist_ok=True)
books_path = os.path.expanduser('~/Edu_AI_Library/data/books_dataset.csv')
users_path = os.path.expanduser('~/Edu_AI_Library/data/users_behavior.csv')

# --- BOOK DATASET GENERATION ---
rows = []
subjects = ["Math", "Science", "Literature", "Environment", "Sports", "Geography"]
languages = ["Arabic", "English"]
availability = ["Available", "Borrowed"]

# 1️⃣ Generate 500 random books
for i in range(500):
    rows.append({
        "id": i,
        "title": f"{random.choice(subjects)} Insights #{i}",
        "author": f"Author {i}",
        "subject": random.choice(subjects),
        "description": f"A book about {random.choice(subjects)} for Qatari students.",
        "year": random.choice(range(1990, 2023)),
        "language": random.choice(languages),
        "availability": random.choice(availability),
        "grade_level": random.choice(range(7, 13))
    })

# 2️⃣ Add guaranteed Arabic History books for Grade 10
for i in range(50):
    rows.append({
        "id": 1000 + i,
        "title": f"Qatar History for Grade 10 #{i}",
        "author": "Ahmad Al-Kuwari",
        "subject": "History",
        "description": "Comprehensive Arabic history book for grade 10 students covering Qatar and Arab civilization.",
        "year": 2018,
        "language": "Arabic",
        "availability": "Available",
        "grade_level": 10
    })

# Write to CSV
with open(books_path, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"✅ Books dataset saved → {books_path}")

# --- USERS BEHAVIOR DATASET ---
users = []
for i in range(200):
    users.append({
        "user_id": f"U{i:03}",
        "borrowed_subject": random.choice(subjects),
        "preferred_language": random.choice(languages),
        "borrow_count": random.randint(1, 15)
    })

with open(users_path, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(users[0].keys()))
    writer.writeheader()
    writer.writerows(users)
print(f"✅ Users dataset saved → {users_path}")

