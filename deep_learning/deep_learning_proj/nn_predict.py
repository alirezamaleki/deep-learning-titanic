def nn_prediction_model(sex, age, sibsp, parch, ticket, fare, cabin, embarked):
  import numpy as np
  from keras.models import load_model
  
  model_predict = load_model('titanic_NN.h5')
  # inputs to the model should be in numpy array format
  x = [[sex, age, sibsp, parch, ticket, fare, cabin, embarked]]
  x_example = np.array(x) 
  prediction_num = model_predict.predict(x_example)
  if prediction_num < 0.5:
    prediction = 'Not survived'
  elif prediction_num > 0.5:
    prediction = 'Survived!'
  else:
    prediction = 'Error'
  return prediction_num


