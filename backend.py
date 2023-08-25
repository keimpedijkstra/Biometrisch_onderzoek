__author__ = "Keimpe Dijkstra"

import random as r
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


class Character_choice:

    def __init__(self, choice):
        self.choice = choice

    def set_choice(self, choice):
        self.choice = choice

    def get_choice(self):
        return self.choice
    

class models:

    def __init__(self):
        self.dummy_data = 'dummydata.csv'

    def dummy_data_gen(self):
        x =0
        with open(self.dummy_data,'w') as file:
            file.write("height,heartrate,haircolor,hoursleep,hoursports,choice,age\n")
            while x < 100:
                height = round(r.gauss(170, 7))
                heartrate = r.randint(60, 100)
                haircolor = r.randint(1,3)
                hoursleep = r.randint(4,10)
                hoursports = r.randint(1,5)
                choice = r.randint(0,1)
                age = r.randint(5,80)
                datapoint = str(height) + "," + str(heartrate) + "," + str(haircolor) + "," + str(hoursleep) + "," + str(hoursports) + "," + str(choice) + "," + str(age)
                file.write(datapoint)
                file.write("\n")
                x+=1

    def read_data(self):
        df = pd.read_csv(self.dummy_data, header=0)
        x = df.drop("age", axis=1)
        y = df["age"]
        X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=125)
        return X_train, X_test, y_train, y_test, df

    def naive_bayes(self, X_train, y_train):
        model = GaussianNB()
        model.fit(X_train.values, y_train.values)
        return model

    def model_predict(self, instance, model):
        return model.predict(instance)

    def linear_regression(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def random_forest(self, X_train, y_train):
        model = RandomForestClassifier()
        model.fit(X_train.values, y_train.values)
        return model

    def model_factory(self, instance):
        instance = np.fromstring(instance, sep=",").reshape(-1,6)
        X_train, X_test, y_train, y_test, df = self.read_data()

        model = self.naive_bayes(X_train=X_train, y_train=y_train)
        prediction_naive_bayes = self.model_predict(model=model, instance=instance)[0]

        model = self.linear_regression(X_train=X_train, y_train=y_train)
        prediction_linear_regression = round(self.model_predict(model=model, instance=instance)[0])

        model = self.random_forest(X_train=X_train, y_train=y_train)
        prediction_random_forest = self.model_predict(model=model, instance=instance)[0]

        return prediction_naive_bayes, prediction_linear_regression, prediction_random_forest
        
class Visualization:
    
    def barplot(data):
        data = pd.read_csv('dummydata.csv')
    
        c = ["Man", "Vrouw"]
        d = [data["choice"].value_counts().to_list()[0], data["choice"].value_counts().to_list()[1]]
        fig = plt.figure(figsize = (10, 5), facecolor='#379CB3')
        ax = plt.axes()
        plt.bar(c,d,color="#246473", width=0.8)
        
        ax.set_facecolor("#379CB3")
        ax.spines['bottom'].set_color('#fff')
        ax.spines['top'].set_color('#fff') 
        ax.spines['right'].set_color('#fff')
        ax.spines['left'].set_color('#fff')
        ax.tick_params(axis='x', colors='#fff')
        ax.tick_params(axis='y', colors='#fff')
        ax.yaxis.label.set_color('#fff')
        ax.xaxis.label.set_color('#fff')
        ax.title.set_color('#fff')

        plt.xlabel("Genders")
        plt.ylabel("Aantal deelnemer")
        plt.title("Distributie mannen en vrouwen")

        plt.savefig("./static/barplot.png")   
    
    def boxplot(data):
        data = pd.read_csv('dummydata.csv')
        d1 = data["hoursleep"].to_list()
        d2 = data["hoursports"].to_list()
        ad = [d1,d2]

        fig = plt.figure(figsize = (10, 5), facecolor='#379CB3')

        ax = plt.axes()
        ax.boxplot(ad, patch_artist=True, notch=True,
            boxprops=dict(facecolor="#246473", color="#246473"),
            capprops=dict(color="#246473"),
            whiskerprops=dict(color="#246473"),
            flierprops=dict(color="#246473", markeredgecolor="#246473"),
            medianprops=dict(color="#246473"))

        ax.set_facecolor("#379CB3")
        ax.spines['bottom'].set_color('#fff')
        ax.spines['top'].set_color('#fff') 
        ax.spines['right'].set_color('#fff')
        ax.spines['left'].set_color('#fff')
        ax.tick_params(axis='x', colors='#fff')
        ax.tick_params(axis='y', colors='#fff')
        ax.yaxis.label.set_color('#fff')
        ax.xaxis.label.set_color('#fff')
        ax.title.set_color('#fff')
        ax.set_xticklabels(["Uur slaap", "Uur sport"])

        plt.ylabel("Aantal uur")
        plt.title("Aantal uur slaap en sport")
        plt.savefig("./static/boxplot")

    def piechart(data):
        
        data = data["haircolor"].to_list()
        d = [data.count(1),data.count(2),data.count(3)]

        fig = plt.figure(figsize = (10, 5), facecolor='#379CB3')

        ax = plt.axes()
        ax.axis('equal')
        
        ax.pie(d,autopct='%1.2f%%', colors=["#debe99", "#4f1a00", "#9a3300"], textprops={'color':"w"})
        ax.title.set_color('#fff')

        plt.title("Distributie haarkleuren")
        
        plt.savefig("./static/piechart")

    def boxplot2(data):
        
        ad = [data["height"].to_list(),data["age"].to_list()]

        fig = plt.figure(figsize = (10, 5), facecolor='#379CB3')

        ax = plt.axes()
        ax.boxplot(ad, patch_artist=True, notch=True,
            boxprops=dict(facecolor="#246473", color="#246473"),
            capprops=dict(color="#246473"),
            whiskerprops=dict(color="#246473"),
            flierprops=dict(color="#246473", markeredgecolor="#246473"),
            medianprops=dict(color="#246473"))

        ax.set_facecolor("#379CB3")
        ax.spines['bottom'].set_color('#fff')
        ax.spines['top'].set_color('#fff') 
        ax.spines['right'].set_color('#fff')
        ax.spines['left'].set_color('#fff')
        ax.tick_params(axis='x', colors='#fff')
        ax.tick_params(axis='y', colors='#fff')
        ax.yaxis.label.set_color('#fff')
        ax.xaxis.label.set_color('#fff')
        ax.title.set_color('#fff')
        ax.set_xticklabels(["Lengte", "Leeftijd"])

        plt.ylabel("cm V jaren")
        plt.title("Lengte en leeftijd")
        plt.savefig("./static/boxplot2")

    def visfactory():
        data = pd.read_csv('dummydata.csv')
        Visualization.barplot(data)
        Visualization.boxplot(data)
        Visualization.piechart(data)
        Visualization.boxplot2(data)