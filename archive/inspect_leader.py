import os
from dotenv import load_dotenv
from lerobot.teleoperators.so101_leader import SO101Leader

print("Inspecting SO101Leader...")
print(dir(SO101Leader))

try:
    leader = SO101Leader(None) # This might fail if config is needed
except:
    pass
