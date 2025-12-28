import os

if __name__ == '__main__':
    print("Welcome to RoboSpeaker 1.1. Created by Palak Sindhi")

    while True:
        x = input("Enter what you want to speak (type 'exit' to quit): ")

        if x.lower() == "exit":
            os.system('powershell -Command '
            '"Add-Type -AssemblyName System.Speech; '
            '(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('
            '\'byee byee dear friend\''
            ')"')
            break

        command = (
            'powershell -Command '
            '"Add-Type -AssemblyName System.Speech; '
            '(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('
            f'\\\"{x}\\\"'
            ')"'
        )

        os.system(command)