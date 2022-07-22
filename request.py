import requests

response = requests.post("https://udacity-ml-project-3.herokuapp.com/predict",json={"model_path": "model/model.pkl",
  "input_data": "{'age': 44,'workclass': 'Private','fnlgt': 326232,'education': 'Bachelors','education-num': 13,'marital-status': 'Divorced','occupation': 'Exec-managerial','relationship': 'Unmarried','race': 'White','sex': 'Male','capital-gain': 0,'capital-loss': 2547,'hours-per-week': 50,'native-country': 'United-States','salary': '>50K'}"})

print(response.status_code)
print(response.json())
