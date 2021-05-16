To choose twitter accounts to train on, run authorSelect.py, and choose between premade list or add own selection. Alternatively don't run this file and use the data i have included. I recommend this option as tweepy has a tendency to lose connection some times. 

The project was mostly ran in Google Colab (for increased processing power) so if any errors occur when running locally i recommend trying it. If Google Colab is used, click "Runtime" -> "Change runtime type" -> "Hardware accelerator" -> "GPU". The project can run without GPU but takes a very long time. If running in google colab run !pip install pytorch-pretrained-bert pytorch-nlp before running BERT.py

To run Naive Bayes, run script.py

To run BERT, run BERT.py