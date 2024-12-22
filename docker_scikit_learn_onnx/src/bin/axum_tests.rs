use axum::{
    routing::post,
    Router,
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
    extract::State,
};
use ndarray::Array2;
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize)]
struct PredictionRequest {
    features: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct PredictionResponse {
    predictions: Vec<f32>,
}

#[derive(Debug)]
enum ModelError {
    PredictionError(String),
    InputError(String),
    OrtError(ort::Error),  
}

// Implement conversion from ort::Error to ModelError
impl From<ort::Error> for ModelError {
    fn from(error: ort::Error) -> Self {
        ModelError::OrtError(error)
    }
}

impl IntoResponse for ModelError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ModelError::PredictionError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ModelError::InputError(msg) => (StatusCode::BAD_REQUEST, msg),
            ModelError::OrtError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };
        
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

struct AppState {
    session: Session,
    input_name: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("toy_model_generation/rf_model.onnx")?;
    
    let input_name = session.inputs[0].name.clone();
    
    let state = Arc::new(AppState {
        session,
        input_name,
    });

    let app = Router::new()
        .route("/predict", post(predict))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    tracing::info!("ML Server running on http://127.0.0.1:3000");
    
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn predict(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PredictionRequest>,
) -> Result<Json<PredictionResponse>, ModelError> {
    if request.features.is_empty() {
        return Err(ModelError::InputError("No features provided".to_string()));
    }
    
    let n_samples = request.features.len();
    let n_features = request.features[0].len();
    
    if n_features != 4 {
        return Err(ModelError::InputError(
            "Each sample must have exactly 4 features".to_string()
        ));
    }
    
    let flat_features: Vec<f32> = request.features.iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    
    let input_array = Array2::from_shape_vec(
        (n_samples, n_features),
        flat_features,
    ).map_err(|e| ModelError::InputError(e.to_string()))?;
    
    // Now the ? operator will work because we implemented From<ort::Error>
    let outputs = state.session
        .run(ort::inputs![&state.input_name => input_array]?)?;
    
    let output = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| ModelError::PredictionError(e.to_string()))?;
    
    let predictions = output.view().iter().copied().collect();
    
    Ok(Json(PredictionResponse { predictions }))
}