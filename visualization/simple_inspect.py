#!/usr/bin/env python3
import pickle

# Load one graph file
with open("connection_graphs/restaurant_full_graph.pkl", "rb") as f:
    data = pickle.load(f)

print("Data type:", type(data))
print("Keys:", list(data.keys()) if isinstance(data, dict) else "Not a dict")

if isinstance(data, dict):
    for key, value in data.items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if isinstance(value, dict):
            print(f"  Length: {len(value)}")
            if len(value) < 10:
                print(f"  Content: {value}")
            else:
                print(f"  First 3 items: {dict(list(value.items())[:3])}")
        else:
            print(f"  Value: {value}") 