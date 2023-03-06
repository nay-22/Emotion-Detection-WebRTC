emotion_model_json = open('emotion_model.json', 'r')
loaded_model = emotion_model_json.read()
emotion_model_json.close()

print(loaded_model)