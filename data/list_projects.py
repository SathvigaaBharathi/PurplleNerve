import os
import json
import urllib.parse

dir_path = r"C:\Users\sathv\.gemini\config\projects"
if os.path.exists(dir_path):
    files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
    for f in files:
        try:
            with open(os.path.join(dir_path, f), "r") as file:
                data = json.load(file)
                name = data.get("name", "Unnamed")
                resources = data.get("projectResources", {}).get("resources", [])
                path = "N/A"
                if resources:
                    uri = resources[0].get("gitFolder", {}).get("folderUri", "")
                    path = urllib.parse.unquote(uri)
                print(f"- {name} ({path})")
        except Exception as e:
            print(f"Error reading {f}: {e}")
