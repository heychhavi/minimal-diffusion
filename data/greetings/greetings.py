# list of 1000 greetings in Python
greetings = [
   "I have faith that things are going smoothly for you and yours|2.58",
"long time no see!|3.425",
"nice to see you too!|3.21",
"I'm happy to see you!|3.464",
"it was so good meeting you|2.496"
]

import pandas as pd
data = pd.DataFrame({"greeting": greetings})
data = data.sample(frac=1).reset_index(drop=True)
data.to_csv("greetings.txt", index=False)
