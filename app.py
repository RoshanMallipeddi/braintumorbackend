import os
from dotenv import load_dotenv
import bcrypt
import tensorflow as tf
import numpy as np
import cv2  # type: ignore
from flask import Flask, request, jsonify, send_from_directory
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime  # Import datetime
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import load_img, img_to_array  # type: ignore
from flask_sqlalchemy import SQLAlchemy  # type: ignore
from flask_migrate import Migrate  # type: ignore # Import Flask-Migrate

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for all domains on all routes
# CORS(app)
# CORS(app, resources={r"/*": {"origins": "*"}}, expose_headers=["Authorization"], allow_headers=["Authorization", "Content-Type"])


# Updated CORS Configuration
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers="*",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    expose_headers=["Authorization"]
)


# Configure database dynamically
if os.getenv("DATABASE_URL"):  # For Heroku deployment
    database_url = os.getenv("DATABASE_URL")
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)  # SQLAlchemy compatibility
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')  # Localhost

# Other configurations
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')  # e.g., 'uploads/mri_images'
app.config['GRADCAM_FOLDER'] = os.getenv('GRADCAM_FOLDER')  # e.g., 'uploads/gradcam_images'
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH'))

# Create upload directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRADCAM_FOLDER'], exist_ok=True)

# Initialize Flask extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)  # Initialize Flask-Migrate
jwt = JWTManager(app)

# Load models
try:
    custom_cnn = load_model('models/custom_cnn_model.h5')
    resnet50_model = load_model('models/resnet50_model.h5')
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Utility functions
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(stored_hash: str, provided_password: str) -> bool:
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_hash.encode('utf-8'))

def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def generate_gradcam(model, img_path, class_idx):
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if not last_conv_layer_name:
        raise ValueError("No Conv2D layer found in the model.")

    img = preprocess_image(img_path)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)

    if isinstance(heatmap, tf.Tensor):
        heatmap = heatmap.numpy()

    return heatmap

def overlay_gradcam(heatmap, img_path, alpha=0.4, output_filename=None):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at path: {img_path}")
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    if not output_filename:
        output_filename = f"gradcam_{os.path.basename(img_path)}"
    output_path = os.path.join(app.config['GRADCAM_FOLDER'], output_filename)
    cv2.imwrite(output_path, superimposed_img)
    return output_path

# Routes for serving uploaded images
@app.route('/uploads/mri_images/<path:filename>', methods=['GET'])
def serve_mri_images(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/gradcam_images/<path:filename>', methods=['GET'])
def serve_gradcam_images(filename):
    return send_from_directory(app.config['GRADCAM_FOLDER'], filename)

# JWT Error Handlers
@jwt.unauthorized_loader
def unauthorized_response(callback):
    return jsonify({'message': 'Missing Authorization Header'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(callback):
    return jsonify({'message': 'Invalid Token'}), 422

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'message': 'Token has expired'}), 401

# Database Models
class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(50), nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    specialization = db.Column(db.String(100))
    license_number = db.Column(db.String(100))

    reports = db.relationship('Report', backref='user', lazy=True)

class Report(db.Model):
    __tablename__ = 'reports'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    cnn_results = db.Column(db.String(200))
    cnn_confidence = db.Column(db.JSON)
    resnet_results = db.Column(db.String(200))
    resnet_confidence = db.Column(db.JSON)
    doctor_notes = db.Column(db.Text)
    gradcam_cnn_path = db.Column(db.String(200))
    gradcam_resnet_path = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, nullable=False)  # Changed to local time
    patient_username = db.Column(db.String(100))

# Home Route
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Brain Tumor Classification System API is running'}), 200

# User Registration
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    role = data.get('role')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')

    if not all([role, first_name, last_name, email, username, password]):
        return jsonify({'message': 'All fields are required.'}), 400

    if role not in ['patient', 'doctor']:
        return jsonify({'message': 'Invalid role specified.'}), 400

    existing_user_email = User.query.filter_by(email=email).first()
    if existing_user_email:
        return jsonify({'message': 'Email already exists.'}), 400

    existing_user_username = User.query.filter_by(username=username).first()
    if existing_user_username:
        return jsonify({'message': 'Username already exists.'}), 400

    user_data = {
        'role': role,
        'first_name': first_name,
        'last_name': last_name,
        'email': email,
        'username': username,
        'password': hash_password(password),
        'created_at': datetime.now()  # Changed to local time
    }

    if role == 'doctor':
        specialization = data.get('specialization')
        license_number = data.get('license_number')
        if not all([specialization, license_number]):
            return jsonify({'message': 'Specialization and License Number are required for doctors.'}), 400
        user_data['specialization'] = specialization
        user_data['license_number'] = license_number

    try:
        new_user = User(**user_data)
        db.session.add(new_user)
        db.session.commit()
    except Exception as e:
        return jsonify({'message': f'Error creating user: {str(e)}'}), 500

    access_token = create_access_token(identity=str(new_user.id))
    return jsonify({'token': access_token, 'role': role, 'first_name': first_name, 'last_name': last_name}), 201

# User Login
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    role = data.get('role')
    email = data.get('email')
    password = data.get('password')
    license_number = data.get('license_number')  # Optional for doctors

    if not all([role, email, password]):
        return jsonify({'message': 'Role, email, and password are required.'}), 400

    if role not in ['patient', 'doctor']:
        return jsonify({'message': 'Invalid role specified.'}), 400

    user = User.query.filter_by(email=email, role=role).first()
    if not user:
        return jsonify({'message': 'Invalid credentials.'}), 401

    if not verify_password(user.password, password):
        return jsonify({'message': 'Invalid credentials.'}), 401

    if role == 'doctor':
        if not license_number:
            return jsonify({'message': 'Medical License Number is required for doctors.'}), 400
        if user.license_number != license_number:
            return jsonify({'message': 'Invalid Medical License Number.'}), 401

    access_token = create_access_token(identity=str(user.id))
    return jsonify({
        'token': access_token,
        'role': role,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'email': user.email
    }), 200

# Upload MRI and Generate Grad-CAM
@app.route('/api/upload', methods=['POST'])
@jwt_required()
def upload_mri():
    if 'mri_image' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['mri_image']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Predict using Custom CNN
            cnn_preds = custom_cnn.predict(preprocess_image(file_path))[0]
            cnn_class = np.argmax(cnn_preds)
            cnn_confidence = {class_labels[i]: f"{cnn_preds[i]*100:.2f}%" for i in range(len(class_labels))}
            cnn_result = class_labels[cnn_class]

            # Predict using ResNet50
            resnet_preds = resnet50_model.predict(preprocess_image(file_path))[0]
            resnet_class = np.argmax(resnet_preds)
            resnet_confidence = {class_labels[i]: f"{resnet_preds[i]*100:.2f}%" for i in range(len(class_labels))}
            resnet_result = class_labels[resnet_class]

            # Generate Grad-CAM for Custom CNN
            heatmap_cnn = generate_gradcam(custom_cnn, file_path, cnn_class)
            gradcam_cnn_path = overlay_gradcam(heatmap_cnn, file_path, alpha=0.4, output_filename=f"gradcam_cnn_{filename}")

            # Generate Grad-CAM for ResNet50
            heatmap_resnet = generate_gradcam(resnet50_model, file_path, resnet_class)
            gradcam_resnet_path = overlay_gradcam(heatmap_resnet, file_path, alpha=0.4, output_filename=f"gradcam_resnet_{filename}")

            # Get current user
            current_user_id = int(get_jwt_identity())
            current_user = User.query.get(current_user_id)
            if not current_user:
                return jsonify({'message': 'User not found.'}), 404
            patient_username = current_user.username

            # Create new report
            new_report = Report(
                user_id=current_user.id,
                filename=filename,
                cnn_results=cnn_result,
                cnn_confidence=cnn_confidence,
                resnet_results=resnet_result,
                resnet_confidence=resnet_confidence,
                doctor_notes='',
                gradcam_cnn_path=gradcam_cnn_path.replace("\\", "/"),
                gradcam_resnet_path=gradcam_resnet_path.replace("\\", "/"),
                created_at=datetime.now(),  # Changed to local time
                patient_username=patient_username
            )
            db.session.add(new_report)
            db.session.commit()

            report = {
                'id': str(new_report.id),
                'filename': new_report.filename,
                'cnn_results': new_report.cnn_results,
                'cnn_confidence': new_report.cnn_confidence,
                'resnet_results': new_report.resnet_results,
                'resnet_confidence': new_report.resnet_confidence,
                'doctor_notes': new_report.doctor_notes,
                'gradcam_cnn_path': new_report.gradcam_cnn_path,
                'gradcam_resnet_path': new_report.gradcam_resnet_path,
                'created_at': new_report.created_at.isoformat(),
                'patient_username': new_report.patient_username
            }

            return jsonify({'message': 'Upload Successful', 'report': report}), 201

        except Exception as e:
            return jsonify({'message': f'Error processing the image: {str(e)}'}), 500

# Get All Reports
@app.route('/api/reports', methods=['GET'])
@jwt_required()
def get_reports():
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'message': 'User not found.'}), 404

    search_username = request.args.get('username')

    if user.role == 'patient':
        reports = Report.query.filter_by(user_id=user.id).all()
    elif user.role == 'doctor':
        if search_username:
            patient = User.query.filter_by(username=search_username, role='patient').first()
            if patient:
                reports = Report.query.filter_by(user_id=patient.id).all()
            else:
                return jsonify({'reports': []}), 200
        else:
            reports = Report.query.all()
    else:
        return jsonify({'message': 'Invalid role.'}), 400

    reports_list = []
    for report in reports:
        report_data = {
            'id': str(report.id),
            'filename': report.filename,
            'cnn_results': report.cnn_results,
            'cnn_confidence': report.cnn_confidence,
            'resnet_results': report.resnet_results,
            'resnet_confidence': report.resnet_confidence,
            'doctor_notes': report.doctor_notes,
            'gradcam_cnn_path': report.gradcam_cnn_path,
            'gradcam_resnet_path': report.gradcam_resnet_path,
            'created_at': report.created_at.isoformat(),
            'patient_username': report.patient_username
        }
        reports_list.append(report_data)

    return jsonify({'reports': reports_list}), 200

# Get Single Report
@app.route('/api/reports/<int:report_id>', methods=['GET'])
@jwt_required()
def get_report(report_id):
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'message': 'User not found.'}), 404

    report = Report.query.get(report_id)
    if not report:
        return jsonify({'message': 'Report not found.'}), 404

    if user.role == 'patient' and report.user_id != user.id:
        return jsonify({'message': 'Unauthorized access to this report.'}), 403

    report_data = {
        'id': str(report.id),
        'filename': report.filename,
        'cnn_results': report.cnn_results,
        'cnn_confidence': report.cnn_confidence,
        'resnet_results': report.resnet_results,
        'resnet_confidence': report.resnet_confidence,
        'doctor_notes': report.doctor_notes,
        'gradcam_cnn_path': report.gradcam_cnn_path,
        'gradcam_resnet_path': report.gradcam_resnet_path,
        'created_at': report.created_at.isoformat(),
        'patient_username': report.patient_username
    }

    return jsonify({'report': report_data}), 200

# Update Report (Doctor Only)
@app.route('/api/reports/<int:report_id>', methods=['PUT'])
@jwt_required()
def update_report(report_id):
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    if not user or user.role != 'doctor':
        return jsonify({'message': 'Unauthorized.'}), 403

    data = request.get_json()
    notes = data.get('doctor_notes', '')

    report = Report.query.get(report_id)
    if not report:
        return jsonify({'message': 'Report not found.'}), 404

    report.doctor_notes = notes
    db.session.commit()

    return jsonify({'message': 'Report notes updated successfully.'}), 200

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

