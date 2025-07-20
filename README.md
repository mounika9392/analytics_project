# ✈️ Flight Passenger Satisfaction Prediction

This project is a web application that predicts whether an airline passenger is **satisfied** or **dissatisfied** based on flight and service details. It uses a trained machine learning model (Random Forest Classifier) built with **scikit-learn** and a user interface powered by **Streamlit**.

---

## 📁 Project Structure

```
analytics_project/
├── app.py                       # Streamlit app
├── satisfaction_prediction.py   # Model training and encoder saving
├── model.pkl                    # Trained ML model
├── label_encoders.pkl           # Saved label encoders for categorical inputs
├── train.csv                    # Dataset used for training
├── test.csv                     # Dataset used for testing
└── README.md                    # Project documentation
```

---

## 🚀 Features

- Predict passenger satisfaction based on flight and service attributes.
- Clean and simple user interface using Streamlit.
- Supports categorical input encoding using LabelEncoders.
- Handles missing features and ensures model compatibility.

---

## 📊 Model Details

- **Algorithm**: Random Forest Classifier  
- **Input Features**:  
  - Demographic: Gender, Age  
  - Flight details: Class, Type of Travel, Flight Distance, Departure/Arrival Delay  
  - Services: Inflight WiFi, Cleanliness, Check-in Service, Baggage Handling, etc.

- **Target**: Passenger Satisfaction (Satisfied / Neutral or Dissatisfied)

---

## 🛠️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/flight-satisfaction-app.git
cd flight-satisfaction-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas scikit-learn streamlit
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧠 Model Training (Optional)

If you want to retrain the model:

```bash
python satisfaction_prediction.py
```

This script:
- Loads `train.csv`
- Preprocesses the data
- Trains the model
- Saves `model.pkl` and `label_encoders.pkl`

---

## 📂 Files to Include for Deployment

Make sure these files are in your project root:
- `app.py`
- `model.pkl`
- `label_encoders.pkl`

---

## 🖼️ User Interface Example

The app collects input for:
- Demographics (Age, Gender)
- Flight details (Distance, Delay)
- Services (WiFi, Cleanliness, etc.)

Then predicts:
> ✅ **Satisfied**  
or  
> ❌ **Neutral or Dissatisfied**

---

## 🧾 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Developed by **Suri Puri**  
Feel free to reach out for collaboration or feedback!
