import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import plotly.express as px

# Year 1
df1 = pd.DataFrame({
    'City': ['New Delhi', 'Dwarka', 'Rohini', 'Karol Bagh', 'Laxmi Nagar'],
    'January': [325, 308, 319, 330, 313],
    'February': [302, 287, 296, 312, 291],
    'March': [287, 272, 281, 297, 277],
    'April': [272, 257, 267, 282, 262],
    'May': [258, 243, 251, 268, 248],
    'June': [242, 227, 237, 253, 233],
    'July': [207, 192, 197, 217, 202],
    'August': [197, 182, 187, 207, 192],
    'September': [223, 208, 218, 233, 219],
    'October': [309, 289, 293, 304, 283],
    'November': [364, 352, 357, 369, 348],
    'December': [384, 372, 377, 387, 367]
})

# DataFrame 2 (values incremented by 5)
df2 = pd.DataFrame({
    'City': ['New Delhi', 'Dwarka', 'Rohini', 'Karol Bagh', 'Laxmi Nagar'],
    'January': [330, 313, 324, 335, 318],
    'February': [307, 292, 301, 317, 296],
    'March': [292, 277, 286, 302, 282],
    'April': [277, 262, 272, 287, 267],
    'May': [263, 248, 256, 273, 253],
    'June': [247, 232, 242, 258, 238],
    'July': [212, 197, 202, 222, 207],
    'August': [202, 187, 192, 212, 197],
    'September': [228, 213, 223, 238, 224],
    'October': [314, 294, 298, 309, 288],
    'November': [369, 357, 362, 374, 353],
    'December': [389, 377, 382, 392, 372]
})

# DataFrame 3 (values incremented by 5)
df3 = pd.DataFrame({
    'City': ['New Delhi', 'Dwarka', 'Rohini', 'Karol Bagh', 'Laxmi Nagar'],
    'January': [335, 318, 329, 340, 323],
    'February': [312, 297, 306, 322, 301],
    'March': [297, 282, 291, 307, 287],
    'April': [282, 267, 277, 292, 272],
    'May': [268, 253, 261, 278, 258],
    'June': [252, 237, 247, 263, 243],
    'July': [217, 202, 207, 227, 212],
    'August': [207, 192, 197, 217, 202],
    'September': [233, 218, 228, 243, 229],
    'October': [319, 299, 303, 314, 293],
    'November': [374, 362, 367, 379, 358],
    'December': [394, 382, 387, 397, 377]
})

# DataFrame 4 (values incremented by 5)
df4 = pd.DataFrame({
    'City': ['New Delhi', 'Dwarka', 'Rohini', 'Karol Bagh', 'Laxmi Nagar'],
    'January': [340, 323, 334, 345, 328],
    'February': [317, 302, 311, 327, 306],
    'March': [302, 287, 296, 312, 292],
    'April': [287, 272, 282, 297, 277],
    'May': [273, 258, 266, 283, 263],
    'June': [257, 242, 252, 268, 248],
    'July': [222, 207, 212, 232, 217],
    'August': [212, 197, 202, 222, 207],
    'September': [238, 223, 233, 248, 234],
    'October': [324, 304, 308, 319, 298],
    'November': [379, 367, 372, 384, 363],
    'December': [399, 387, 392, 402, 382]
})

# DataFrame 5 (values incremented by 5)
df5 = pd.DataFrame({
    'City': ['New Delhi', 'Dwarka', 'Rohini', 'Karol Bagh', 'Laxmi Nagar'],
    'January': [345, 328, 339, 350, 333],
    'February': [322, 307, 316, 332, 311],
    'March': [307, 292, 301, 317, 297],
    'April': [292, 277, 287, 302, 282],
    'May': [278, 263, 271, 288, 268],
    'June': [262, 247, 257, 273, 253],
    'July': [227, 212, 217, 237, 222],
    'August': [217, 202, 207, 227, 212],
    'September': [243, 228, 238, 253, 239],
    'October': [329, 309, 313, 324, 303],
    'November': [384, 372, 377, 389, 368],
    'December': [404, 392, 397, 407, 387]
})

# DataFrame 6 (values incremented by 5)
df6 = pd.DataFrame({
    'City': ['New Delhi', 'Dwarka', 'Rohini', 'Karol Bagh', 'Laxmi Nagar'],
    'January': [350, 333, 344, 355, 338],
    'February': [327, 312, 321, 337, 316],
    'March': [312, 297, 306, 322, 302],
    'April': [297, 282, 292, 307, 287],
    'May': [283, 268, 276, 293, 273],
    'June': [267, 252, 262, 278, 258],
    'July': [232, 217, 222, 242, 227],
    'August': [222, 207, 212, 232, 217],
    'September': [248, 233, 243, 258, 244],
    'October': [334, 314, 318, 329, 308],
    'November': [389, 377, 382, 394, 373],
    'December': [409, 397, 402, 412, 392]
})

# DataFrame 7 (values incremented by 5)
df7 = pd.DataFrame({
    'City': ['New Delhi', 'Dwarka', 'Rohini', 'Karol Bagh', 'Laxmi Nagar'],
    'January': [355, 338, 349, 360, 343],
    'February': [332, 317, 326, 342, 321],
    'March': [317, 302, 311, 327, 307],
    'April': [302, 287, 297, 312, 292],
    'May': [288, 273, 281, 298, 278],
    'June': [272, 257, 267, 283, 263],
    'July': [237, 222, 227, 247, 232],
    'August': [227, 212, 217, 237, 222],
    'September': [253, 238, 248, 263, 249],
    'October': [339, 319, 323, 334, 313],
    'November': [394, 382, 387, 399, 378],
    'December': [414, 402, 407, 417, 397]
})

# DataFrame 8 (values incremented by 5)
df8 = pd.DataFrame({
    'City': ['New Delhi', 'Dwarka', 'Rohini', 'Karol Bagh', 'Laxmi Nagar'],
    'January': [360, 343, 354, 365, 348],
    'February': [337, 322, 331, 347, 326],
    'March': [322, 307, 316, 332, 312],
    'April': [307, 292, 302, 317, 297],
    'May': [293, 278, 286, 303, 283],
    'June': [277, 262, 272, 288, 268],
    'July': [242, 227, 232, 252, 237],
    'August': [232, 217, 222, 242, 227],
    'September': [258, 243, 253, 268, 254],
    'October': [344, 324, 328, 339, 318],
    'November': [399, 387, 392, 404, 383],
    'December': [419, 407, 412, 422, 402]
})

# DataFrame 9 (values incremented by 5)
df9 = pd.DataFrame({
    'City': ['New Delhi', 'Dwarka', 'Rohini', 'Karol Bagh', 'Laxmi Nagar'],
    'January': [365, 348, 359, 370, 353],
    'February': [342, 327, 336, 352, 331],
    'March': [327, 312, 321, 337, 317],
    'April': [312, 297, 307, 322, 302],
    'May': [298, 283, 291, 308, 288],
    'June': [282, 267, 277, 293, 273],
    'July': [247, 232, 237, 257, 242],
    'August': [237, 222, 227, 247, 232],
    'September': [263, 248, 258, 273, 259],
    'October': [349, 329, 333, 344, 323],
    'November': [404, 392, 397, 409, 388],
    'December': [424, 412, 417, 427, 407]
})

# DataFrame 10 (values incremented by 5)
df10 = pd.DataFrame({
    'City': ['New Delhi', 'Dwarka', 'Rohini', 'Karol Bagh', 'Laxmi Nagar'],
    'January': [370, 353, 364, 375, 358],
    'February': [347, 332, 341, 357, 336],
    'March': [332, 317, 326, 342, 322],
    'April': [317, 302, 312, 327, 307],
    'May': [303, 288, 296, 313, 293],
    'June': [287, 272, 282, 298, 278],
    'July': [252, 237, 242, 262, 247],
    'August': [242, 227, 232, 252, 237],
    'September': [268, 253, 263, 278, 264],
    'October': [354, 334, 338, 349, 328],
    'November': [409, 397, 402, 414, 393],
    'December': [429, 417, 422, 432, 412]
})





list2 = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]
year = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
n = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
X = []

background_css = """
<style>
    body, .stApp {
        background: linear-gradient(to bottom, #E8F5E9, #FFFFFF);
        font-family: Arial, sans-serif;
        color: #2E7D32;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1B5E20;
    }
    
    p, span, div {
        color: #2E7D32;
    }
</style>
"""

st.markdown(background_css, unsafe_allow_html=True)

st.title("Air Shield")

choice = st.selectbox("Select an option:", [None, "AQI Analysis Of Last 10 Years Using HeatMap", "Prediction Of Future AQI Levels In Different Districts"])

if choice == "AQI Analysis Of Last 10 Years Using HeatMap":
    y = st.selectbox("Select an option:", year)
    
    index = year.index(y)
    df = list2[int(index)].copy()  # Create a copy of the DataFrame
    df_heatmap = df.set_index('City')  # Only set index for heatmap visualization

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df_heatmap, annot=False, cmap='YlGnBu', linewidths=0.5, ax=ax)
    ax.set_title(f"AQI Heatmap for Different Months ({y})")
    ax.set_xlabel("Month")
    ax.set_ylabel("City")

    st.pyplot(fig)


elif choice == "Prediction Of Future AQI Levels In Different Districts":
    states = ['New Delhi', 'Dwarka', 'Rohini', 'Karol Bagh', 'Laxmi Nagar']
    choices = st.selectbox("Select an option:", [None] + states)
    
    if choices in states:
        # Calculate mean AQI for each year for the selected city
        for df in list2:
            # Filter row for selected city and calculate mean of all months
            city_data = df[df['City'] == choices].iloc[:, 1:].values.mean()  # Skip 'City' column
            X.append(city_data)
        
        year_nums = np.array(range(2015, 2025)).reshape(-1, 1)
        X = np.array(X)
        
        x_train, x_test, y_train, y_test = train_test_split(year_nums, X, test_size=0.1, random_state=3)
        model = LinearRegression()
        model.fit(x_train, y_train)
        
        predictions = []
        
        for i in range(2025 , 2036):
            future_year = np.array([[i]])
            prediction = model.predict(future_year)[0]
            predictions.append(prediction)

        # Predict for 2025

        
        st.subheader(f"Predicted Average AQI For The Next 10 Years")
        
        f_years = [2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]

        d = pd.DataFrame(list(zip(predictions, f_years)), columns=['Predictions', 'Years'])
        # Optional: Add a line plot to show the trend
        fig = px.line(d, x='Predictions', y='Years', title='Line Graph Of Predicted AQI Of Upcoming Years', markers=True)
        fig.update_layout(xaxis_title="Predictions", yaxis_title="Years")

        # Display the graph in Streamlit
        st.write("Line Graph Of Predicted AQI Of Upcoming Years")
        st.plotly_chart(fig)
