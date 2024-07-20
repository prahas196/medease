from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib

app = Flask(__name__)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Load the trained model and data
model = joblib.load('models/model.pkl')
disease_df = pd.read_csv('Disease.csv')
drug_df = pd.read_csv('Drug.csv')

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    symptoms = db.Column(db.String(200), nullable=False)
    diagnosis = db.Column(db.String(100), nullable=False)
    medicines = db.Column(db.String(200), nullable=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        symptoms = request.form.getlist('symptoms')
        symptoms_dict = {symptom: 1 if symptom in symptoms else 0 for symptom in disease_df.columns if symptom != 'prognosis'}
        symptoms_df = pd.DataFrame([symptoms_dict])
        
        # Predict disease
        disease = model.predict(symptoms_df)[0]
        
        # Recommend medicines
        recommended_drugs = drug_df[drug_df['Disease'] == disease]['Drug'].tolist()
        
        return render_template('index.html', symptoms=disease_df.columns.drop('prognosis'), prognosis=disease, drugs=recommended_drugs)
    
    # Get symptom columns from the disease dataframe
    symptoms = disease_df.columns.drop('prognosis')
    
    return render_template('index.html', symptoms=symptoms)

@app.route('/medicine', methods=['GET', 'POST'])
def medicine():
    recommended_drugs = None
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = request.form['gender']
        
        # Recommend medicines based on age and gender
        recommended_drugs = drug_df[(drug_df['Age'] <= age) & (drug_df['Gender'] == gender)]['Drug'].tolist()
        
    return render_template('medicine.html', drugs=recommended_drugs)

@app.route('/patient', methods=['GET', 'POST'])
def patient():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        symptoms = request.form.getlist('symptoms')
        symptoms_str = ','.join(symptoms)
        
        symptoms_dict = {symptom: 1 if symptom in symptoms else 0 for symptom in disease_df.columns if symptom != 'prognosis'}
        symptoms_df = pd.DataFrame([symptoms_dict])
        diagnosis = model.predict(symptoms_df)[0]
        medicines = drug_df[drug_df['Disease'] == diagnosis]['Drug'].tolist()
        medicines_str = ','.join(medicines)
        
        patient = Patient(name=name, age=age, gender=gender, symptoms=symptoms_str, diagnosis=diagnosis, medicines=medicines_str)
        db.session.add(patient)
        db.session.commit()
        
        return redirect(url_for('doctor'))
    
    symptoms = disease_df.columns.drop('prognosis')
    return render_template('patient.html', symptoms=symptoms)

@app.route('/doctor')
def doctor():
    patients = Patient.query.all()
    return render_template('doctor.html', patients=patients, drug_df=drug_df)

@app.route('/update_medicines/<int:patient_id>', methods=['POST'])
def update_medicines(patient_id):
    patient = Patient.query.get(patient_id)
    selected_medicines = request.form.getlist('medicines')
    patient.medicines = ','.join(selected_medicines)
    db.session.commit()
    return redirect(url_for('doctor'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
