import json
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime

# Load the db_dump data (list of dicts)
with open('db_dump_일반.json', 'r', encoding='utf-8') as f:
    db_data = json.load(f)['일반']

# Load the id2name mapping (dict of user_id to user name)
with open('id2name.json', 'r', encoding='utf-8') as f:
    id2name = json.load(f)

# Ensure keys in id2name are integers
id2name = {int(k): v for k, v in id2name.items()}

# Set your start and end timestamps as ISO formatted strings
start_timestamp_str = "2019-03-18 00:00:00+00:00"  # Change as needed
end_timestamp_str = "2023-09-25 23:59:59+00:00"      # Change as needed

# Convert timestamp strings to datetime objects
start_timestamp = datetime.fromisoformat(start_timestamp_str)
end_timestamp = datetime.fromisoformat(end_timestamp_str)

# Filter messages that occurred between the start and end timestamps (inclusive)
filtered_messages = [
    msg for msg in db_data 
    if start_timestamp <= datetime.fromisoformat(msg["time"]) <= end_timestamp
]

# Count messages per user in the filtered messages
user_counts = Counter(msg["user_id"] for msg in filtered_messages)
total_messages = sum(user_counts.values())

# Prepare data: compute percentage and filter out users with <1% of messages
data = []
for user_id, count in user_counts.items():
    perc = (count / total_messages) * 100
    if perc < 1:
        continue
    name = id2name.get(user_id, str(user_id))
    data.append((name, perc))

# Sort the data by percentage in descending order
data.sort(key=lambda x: x[1], reverse=True)
names, percentages = zip(*data) if data else ([], [])
print(names)
# Plotting
plt.figure(figsize=(10, 6))
plt.bar(names, percentages)
plt.xlabel("User")
plt.ylabel("Percentage of Chat Messages (%)")
plt.title(f"Chat Messages per User from {start_timestamp_str} to {end_timestamp_str} (Percentage, >1% Only)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot.png")
