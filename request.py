import requests

response = requests.post('https://udacity-ml-project-3.herokuapp.com/predict/')

print(response.status_code)
print(response.json())
