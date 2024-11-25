import streamlit as st
import pickle

with open('final_random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)