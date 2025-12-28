import requests
import json
import win32com.client as speech

city = input("Enter the name of the city: ")

url = f"https://api.weatherapi.com/v1/current.json?key=ce620c5c944f495fbc5181311252312&q={city}"
r = requests.get(url)

wdic = json.loads(r.text)
w = wdic["current"]["temp_c"]

print(f"Temperature in {city}: {w}°C")

# Correct speech object
speak = speech.Dispatch("SAPI.SpVoice")
text = f"The current weather in {city} is {w} degrees Celsius"
speak.Speak(text)

print(f"Temperature in {city}: {w}°C")

# Correct speech object
speak = speech.Dispatch("SAPI.SpVoice")
text = f"The current weather in {city} is {w} degrees Celsius"
speak.Speak(text)

