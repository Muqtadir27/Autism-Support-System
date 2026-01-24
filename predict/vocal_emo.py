import speech_recognition as sr
import pyttsx3
import threading
import time

def vocal_expression_interpretation():
    recognizer = sr.Recognizer()
    # Initialize text-to-speech engine with error handling
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        tts_available = True
    except Exception as e:
        print(f"Text-to-speech not available: {e}")
        tts_available = False

    # Support Responses
    support_responses = {
        'happy': "It's wonderful to hear that you're feeling happy! Keep that positive energy going.",
        'sad': "I hear that you're feeling a bit down. Remember, it's okay to feel this way. I'm here for you.",
        'angry': "I can hear some frustration. Let's take a deep breath together. You're doing great.",
        'fear': "You sound a bit anxious. You are in a safe space. Everything is going to be alright.",
        'neutral': "I'm listening. Feel free to express yourself whenever you're ready."
    }

    # Keywords for basic sentiment
    keywords = {
        'happy': ['happy', 'great', 'good', 'excellent', 'joy', 'wonderful', 'fine', 'fantastic', 'amazing'],
        'sad': ['sad', 'down', 'bad', 'unhappy', 'lonely', 'hurt', 'cry', 'depressed', 'upset'],
        'angry': ['angry', 'mad', 'frustrated', 'hate', 'annoyed', 'furious', 'irritated'],
        'fear': ['scared', 'afraid', 'anxious', 'nervous', 'worry', 'help', 'panic', 'terrified']
    }

    def speak(text):
        print(f"[TTS] Speaking: {text}")
        if tts_available:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
        else:
            print(f"TTS Unavailable - Would have said: {text}")

    print("=" * 60)
    print("VOCAL INTERPRETATION MODULE: ACTIVE")
    print("=" * 60)
    speak("Vocal Support System initialized. I am listening. How are you feeling today?")

    try:
        with sr.Microphone() as source:
            print("[INFO] Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("[INFO] Calibration complete. You may speak now.")
            
            while True:
                print("\n[LISTENING...] Speak clearly into the microphone.")
                try:
                    audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
                    print("[PROCESSING] Analyzing audio input...")
                    text = recognizer.recognize_google(audio).lower()
                    print(f"[DETECTED] You said: '{text}'")

                    if any(word in text for word in ['exit', 'stop', 'terminate', 'quit', 'bye']):
                        speak("Terminating vocal support sequence. Stay safe and take care.")
                        print("[INFO] Session terminated by user.")
                        break

                    detected_emotion = 'neutral'
                    for emotion, words in keywords.items():
                        if any(word in text for word in words):
                            detected_emotion = emotion
                            break
                    
                    response = support_responses.get(detected_emotion)
                    print(f"[EMOTION DETECTED] {detected_emotion.upper()}")
                    print(f"[RESPONSE] {response}")
                    speak(response)

                except sr.WaitTimeoutError:
                    print("[TIMEOUT] No speech detected. Continuing to listen...")
                    continue
                except sr.UnknownValueError:
                    print("[ERROR] Could not understand audio. Please speak more clearly.")
                    speak("I couldn't understand that. Could you please speak more clearly?")
                    continue
                except sr.RequestError as e:
                    print(f"[ERROR] Google Speech Recognition service error: {e}")
                    speak("I'm having trouble connecting to the speech service. Please try again later.")
                    break
                except Exception as e:
                    print(f"[ERROR] Unexpected error: {e}")
                    break
    except OSError as e:
        print(f"[FATAL] Microphone Error: {e}")
        print("[HINT] Please ensure your microphone is properly connected and enabled.")
        speak("Microphone initialization failed. Please check your audio device.")
    except Exception as e:
        print(f"[FATAL] System Error: {e}")
    finally:
        if tts_available:
            try:
                engine.stop()
            except Exception as e:
                print(f"TTS Stop Error: {e}")
        print("=" * 60)
        print("VOCAL INTERPRETATION MODULE: TERMINATED")
        print("=" * 60)
