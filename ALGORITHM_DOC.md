# DDoS Detection and Mitigation Algorithms

## Input (D): Network Traffic Data Stream

**Description:** Real-time network traffic data from monitoring points containing raw flow statistics and packet-level features

---

## Output (R): Threat Classification with Mitigation Actions and Performance Metrics

**Format:** Structured predictions including threat status, confidence scores, severity classification, and recommended mitigation actions

---

## Algorithm 1: Data Preprocessing and Feature Engineering

### Input: Raw CIC-DDoS2019 Dataset (D_raw)

### Output: Preprocessed Dataset (D_processed) ready for training or inference

### 1. Load Network Traffic Data
   **Load CIC-DDoS2019 Dataset:**
   - 85 columns, 225,745 samples
   - Read from CSV file
   - Parse raw packet captures and flow statistics

### 2. Initial Data Inspection
   **Clean column names and values:**
   - Strip whitespace from column names
   - Strip whitespace from string values
   - Convert to absolute values for numerical columns
   
   **Identify column types:**
   - Categorical features (object dtype)
   - Numerical features (int/float)
   - Binary columns (nunique == 2)

### 3. Data Cleaning & Validation
   **Remove redundant/problematic columns:**
   - Drop constant columns (nunique == 1) → ~7 columns
   - Drop duplicate columns (identical values) → ~17 columns
   - Drop ID columns: Flow ID
   - Drop columns with infinity values: Flow Bytes/s, Flow Packets/s
   
   **Handle missing and invalid values:**
   - Replace infinity with NaN
   - Impute missing values with median
   - Remove duplicate rows → 2 rows removed
   
   **Normalize timestamp:**
   - Convert to datetime format
   - Sort by timestamp

### 4. Feature Engineering - Label Encoding
   **Binary Label Encoding:**
   - Map: 'BENIGN' → 0, all attacks → 1
   
   **IP and Port Feature Engineering:**
   - Creates 8 new features from 4 columns:
     - src_is_private - Is source IP in private range?
     - src_is_multicast - Is source IP multicast?
     - dst_is_private - Is destination IP private?
     - dst_is_multicast - Is destination IP multicast?
     - src_port_category - Source port range (well-known/registered/dynamic)
     - dst_port_category - Destination port range
     - dst_is_web_port - Is destination port 80 or 443?
     - dst_is_common_service - Is destination port common service?
   
   **Remove original columns:** Source IP, Source Port, Destination IP, Destination Port

### 5. Correlation Analysis & Feature Selection
   - Drop Timestamp column (not needed for modeling)
   - Calculate correlation matrix
   - Remove highly correlated features (threshold > 0.95)
   
   **Result:** ~38 final features including:
   - Flow statistics (duration, packets, bytes)
   - Packet timing (IAT mean, std, max, min)
   - Flag counts (FIN, SYN, RST, PSH, ACK, URG)
   - Window sizes, header lengths
   - Engineered IP/Port features
   - Binary label

### 6. Data Preprocessing - Scaling
   **Create NetworkDataScaler:**
   - Scaler type: RobustScaler (handles outliers)
   - Alternative options: MinMaxScaler, StandardScaler
   
   **Apply scaling:**
   - Fit scaler on training data only
   - Transform all data splits using fitted scaler
   
   **Result:** D_processed with normalized features ready for training or inference

---

## Algorithm 2: Model Training Algorithm

### Input: Preprocessed Dataset (D_processed from Algorithm 1)

### 1. Split Data into Train/Validation/Test Sets
   **Partition dataset:**
   - Training set: 70% (157,022 samples)
   - Validation set: 15% (33,862 samples)
   - Test set: 15% (33,861 samples)
   
   **Create data batches:**
   - Batch size: 256 samples
   - Shuffle training data for randomization
   - Sequential loading for validation and test sets

### 2. Define Model Architectures
   **Initialize 5 different models with hyperparameters:**
   
   **a) DBNNetwork (Deep Belief Network)**
   - Input dimension: 38
   - Hidden layers: [64, 32]
   - Output dimension: 1 (binary classification)
   - Dropout rate: 0.2
   - Sequence length: 10 time steps
   - Batch normalization: enabled
   
   **b) DDoSTransformer (Attention-based)**
   - Input dimension: 38
   - Model dimension (d_model): 256
   - Attention heads: 8
   - Transformer layers: 4
   - Feedforward dimension: 512
   - Dropout rate: 0.1
   
   **c) LSTMNetwork (Long Short-Term Memory)**
   - Input dimension: 38
   - Hidden dimension: 32
   - LSTM layers: 2
   - Dropout rate: 0.2
   - Bidirectional: False
   
   **d) RNNClassifier (Recurrent Neural Network)**
   - Input size: 38
   - Hidden size: 128
   - RNN layers: 1
   - Dropout rate: 0.3
   - Activation: tanh
   
   **e) TimeSeriesAutoEncoder**
   - Sequence length: 10
   - Feature count: 38
   - Encoding dimensions: [128, 64, 32]
   - Dropout rate: 0.2
   - Batch normalization: enabled

### 3. Set Training Configuration
   - Optimizer: Adam with learning rate 0.0001
   - Loss function: Binary Cross-Entropy with Logits
   - Training epochs: 100
   - Batch size: 256
   - Early stopping patience: 10 epochs
   - Validation frequency: Every epoch

### 4. Training Loop
   **For each epoch (1 to 100):**
   
   **Training Phase:**
   - Set model to training mode
   - For each batch in training set:
     - Forward pass: compute model output
     - Calculate loss: compare output with true labels
     - Backward pass: compute gradients
     - Update weights: apply optimizer step
     - Track training loss
   
   **Validation Phase:**
   - Set model to evaluation mode
   - For each batch in validation set:
     - Forward pass: compute model output (no gradient)
     - Calculate validation loss
     - Generate predictions: apply sigmoid + threshold (0.5)
     - Store predictions and probabilities

### 5. Calculate Validation Metrics
   **After each epoch, compute:**
   - Accuracy: (TP + TN) / Total
   - Precision: TP / (TP + FP)
   - Recall/Sensitivity: TP / (TP + FN)
   - F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
   - Cohen's Kappa: Agreement metric
   - ROC-AUC: Area under ROC curve
   - Validation loss
   
   **Track best performing model:**
   - If current F1-Score > best F1-Score:
     - Update best F1-Score
     - Record best epoch number
     - Save model state (weights and biases)

### 6. Early Stopping
   **Monitor validation performance:**
   - Track consecutive epochs without improvement
   - If no improvement for 10 consecutive epochs:
     - Stop training early
     - Prevent overfitting
     - Return best model from previous epochs

### 7. Save Model Checkpoint
   **After training completes:**
   - Generate timestamp identifier (e.g., 1758192440)
   - Create checkpoint containing:
     - Model class name
     - Best model state (weights and biases)
     - Model architecture configuration
     - Training metrics (best epoch, F1-score, accuracy)
     - Optimizer state
   
   **Save to disk:**
   - Filename format: `{ModelName}_{Timestamp}.pt`
   - Storage location: `models/` directory
   
   **Saved checkpoint files:**
   - DBNNetwork_1758192440.pt
   - DDoSTransformer_1758192530.pt
   - LSTMNetwork_1758192370_minimal.pt
   - RNNClassifier_1758192485.pt
   - TimeSeriesAutoEncoder_1758192664.pt

### 8. Evaluate on Test Set
   **Load best model and perform final evaluation:**
   - Restore best model weights from training
   - Set model to evaluation mode
   - Process test set in batches:
     - Forward pass: generate predictions
     - Apply sigmoid activation
     - Apply threshold (0.5) for binary classification
     - Store predictions and probability scores
   
   **Calculate comprehensive test metrics:**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix: TP, TN, FP, FN counts
   - Sensitivity and Specificity
   - ROC-AUC and PR-AUC scores
   - Cohen's Kappa coefficient

### 9. Save Training History
   **Record training progress to CSV:**
   - Epoch numbers
   - Training loss per epoch
   - Validation loss per epoch
   - Validation accuracy per epoch
   - Validation F1-scores per epoch
   
   **Output location:** `data/{model_name}/training_history.csv`

### 10. Return Training Results
   **Repeat steps 2-9 for all 5 model architectures:**
   - DBNNetwork
   - DDoSTransformer
   - LSTMNetwork
   - RNNClassifier
   - TimeSeriesAutoEncoder
   
   **Final output:** 5 pretrained model checkpoints (*.pt files) ready for ensemble inference

---

## Algorithm 3: DDoS Detection Algorithm (Ensemble Inference)

### Input: Preprocessed Data (D_processed from Algorithm 1) + Pretrained Models (from Algorithm 2)

### 1. Define System Parameters
   - **Attack Types** = {SYN Flood, UDP Flood, HTTP Flood, ICMP Flood, DNS Amplification}
   - **Detection Models M** = {DBN, LSTM, RNN, Transformer, AutoEncoder}
   - **System Threshold** = 0.5 (threat detection boundary)

### 2. Initialize Ensemble Weights
   **Define model ensemble:**
   ```python
   ensemble = ModelEnsemble.from_directory("models", model_indices=[1,2,3,4,5])
   ```
   
   **Models loaded from checkpoints:**
   - **DBNNetwork_1758192440.pt** - Deep Belief Network
   - **DDoSTransformer_1758192530.pt** - Attention-based Transformer
   - **LSTMNetwork_1758192370_minimal.pt** - Long Short-Term Memory
   - **RNNClassifier_1758192485.pt** - Recurrent Neural Network
   - **TimeSeriesAutoEncoder_1758192664.pt** - Unsupervised AutoEncoder
   
   **Weight options:**
   - Equal weights: [0.20, 0.20, 0.20, 0.20, 0.20] (default)
   - Custom weights based on validation performance
   - Weights automatically normalized to sum to 1.0

### 3. Load Pretrained Models
   - Load checkpoint: `torch.load(model_path, map_location=device)`
   - Extract model architecture from checkpoint: `infer_architecture_from_checkpoint()`
   - Instantiate model with correct parameters
   - Load state dict: `model.load_state_dict(checkpoint['model_state_dict'])`
   - Set to evaluation mode: `model.eval()`
   - Move to device (GPU/CPU): `model.to(device)`
   
   **Fallback handling:**
   - If loading fails, try CPU loading
   - Try alternative versions (_ultra_clean, _minimal, _ultra_minimal)
   - Continue with successfully loaded models only

### 4. Generate Individual Predictions
   **Create DataLoader for batch processing:**
   ```python
   test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
   ```
   
   ```python
   with torch.no_grad():
       for inputs, labels in test_loader:
           model_outputs = model(inputs)
           probs = torch.sigmoid(model_outputs).cpu().numpy()
           preds = (probs > 0.5).astype(int)
   ```
   
   - **predictions_m** = Binary predictions (0 or 1)
   - **confidence_m** = Probability scores [0, 1]
   - Store individual predictions and probabilities for each model

### 5. Ensemble Aggregation (Weighted Voting)
   **Ensemble Score Calculation:**
   
   ```python
   weighted_probs = np.zeros_like(batch_predictions[0])
   for i, probs in enumerate(batch_predictions):
       weighted_probs += weights[i] * probs
   ```
   
   **Mathematical formula:**
   
   $$weighted\_score = \sum_{m \in M} (confidence_m \times w_m)$$
   
   Where:
   - $w_m$ = normalized ensemble weight for model m (sum = 1.0)
   - $confidence_m$ ∈ [0, 1] = model probability output
   - Final prediction: weighted_score ≥ 0.5 → 1, else → 0

### 6. Threshold-Based Classification
   **Binary Classification:**
   ```python
   binary_predictions = (weighted_score > 0.5).astype(int)
   ```
   
   **Threat Status Determination:**
   - IF weighted_score ≥ 0.5:
     - threat_status = 1 (THREAT_DETECTED)
     - Use confidence score for mitigation decisions
   - ELSE:
     - threat_status = 0 (BENIGN)
   
   **Severity Classification (for mitigation):**
   - CRITICAL: score ≥ 0.9 → BLOCK
   - HIGH: 0.7 ≤ score < 0.9 → RATE_LIMIT
   - MEDIUM: 0.5 ≤ score < 0.7 → MONITOR
   - LOW: score < 0.5 → ALLOW

### 7. Calculate Comprehensive Metrics
   **Detection Performance Metrics:**
   ```python
   metrics = calculate_comprehensive_metrics(labels, predictions, probabilities)
   ```
   
   **Confusion Matrix:**
   - True Positives (TP): Correctly detected attacks
   - True Negatives (TN): Correctly identified benign traffic
   - False Positives (FP): Benign flagged as attack
   - False Negatives (FN): Missed attacks
   
   **Calculated Metrics:**
   - Accuracy = (TP + TN) / Total
   - Precision = TP / (TP + FP)
   - Recall/Sensitivity = TP / (TP + FN)
   - Specificity = TN / (TN + FP)
   - F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
   - Cohen's Kappa = Agreement accounting for chance
   - ROC-AUC = Area under ROC curve
   - PR-AUC = Precision-Recall AUC

### 8. Generate Detection Report
   **Create results dictionary:**
   ```python
   results = {
       'ensemble_probabilities': weighted_scores,
       'ensemble_predictions': binary_predictions,
       'true_labels': actual_labels,
       'individual_predictions': {model_name: preds for each model},
       'individual_probabilities': {model_name: probs for each model},
       'ensemble_metrics': overall_metrics,
       'individual_metrics': {model_name: metrics for each model},
       'model_names': list of loaded models,
       'weights': normalized weights
   }
   ```

### 9. Save Detection Results
   **Create output folder:** `ensemble_data/{model_combination}/`
   
   **Save 4 CSV files:**
   1. **predictions.csv** - Per-sample predictions
      - sample_id, true_label, ensemble_probability, ensemble_prediction
      - Individual model probabilities and predictions
   
   2. **metrics.csv** - Comprehensive metrics
      - Ensemble metrics (accuracy, F1, precision, recall, etc.)
      - Individual model metrics
   
   3. **config.csv** - Model configuration
      - Model names, paths, weights, load status
   
   4. **summary.csv** - High-level statistics
      - Total samples, positive/negative predictions, key metrics

### 10. Return Output
   **Return R:** Complete results dictionary with detection reports, predictions, metrics, and model information

---

## Algorithm 4: DDoS Mitigation Algorithm

### Input: Ensemble Detection Results from Algorithm 3

### 1. Initialize DDoS Mitigator
   ```python
   mitigator = DDoSMitigator(
       high_confidence_threshold=0.9,
       medium_confidence_threshold=0.7,
   )
   ```

### 2. Apply Mitigation Strategies
   **For each sample with (probability, prediction, true_label):**
   
   ```python
   IF probability ≥ 0.9:  # HIGH threshold
       action = "BLOCK"
       confidence_level = "HIGH"
       # Immediate firewall block
   
   ELSE IF probability ≥ 0.7:  # MEDIUM threshold
       action = "RATE_LIMIT"
       confidence_level = "MEDIUM"
       # Throttle to 100 req/sec

   ELSE:  # Below LOW threshold
       action = "ALLOW"
       confidence_level = "VERY_LOW"
       # Normal traffic flow
   ```

### 3. Determine Action Correctness
   **Check if mitigation action was appropriate:**
   
   ```python
   IF true_label == 1 (actual attack):
       correct_action = action in ['BLOCK', 'RATE_LIMIT', 'MONITOR']
   ELSE (benign traffic):
       correct_action = action in ['ALLOW', 'MONITOR']
   ```
   
   **Classify mitigation outcomes:**
   - TRUE_POSITIVE_MITIGATION: Attack correctly mitigated
   - FALSE_POSITIVE_MITIGATION: Benign traffic incorrectly blocked/limited
   - TRUE_NEGATIVE_MITIGATION: Benign traffic correctly allowed
   - FALSE_NEGATIVE_MITIGATION: Attack incorrectly allowed
   - PARTIAL_MITIGATION: Attack monitored but not blocked

### 4. Create Mitigation Actions DataFrame
   **Store per-sample mitigation data:**
   ```python
   mitigation_actions = pd.DataFrame({
       'sample_id': range(len(probabilities)),
       'probability': ensemble_probabilities,
       'prediction': ensemble_predictions,
       'true_label': actual_labels,
       'action': action_taken,
       'confidence_level': level,
       'is_correct_action': correctness,
       'outcome': mitigation_outcome
   })
   ```

### 5. Calculate Mitigation Performance Metrics
   **Core metrics:**
   - **Mitigation Success Rate** = TP_mitigation / Total_Attacks
   - **False Positive Rate** = FP_mitigation / Total_Benign
   - **Legitimate Traffic Preservation** = TN_mitigation / Total_Benign
   - **Mitigation Accuracy** = Correct_Actions / Total_Samples
   - **Mitigation Precision** = TP_mitigation / (TP + FP)_mitigation
   - **Mitigation Recall** = TP_mitigation / (TP + FN)_mitigation
   - **Mitigation F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
   
   **Action statistics:**
   - Count of BLOCK, RATE_LIMIT, MONITOR, ALLOW actions
   - Average confidence per action type
   - True attacks caught vs benign affected per action

### 6. Generate Strategy Breakdown
   **For each action type {BLOCK, RATE_LIMIT, MONITOR, ALLOW}:**
   ```python
   breakdown[action] = {
       'count': number of samples,
       'percentage': percentage of total,
       'correct_actions': correctly applied,
       'accuracy': action accuracy,
       'avg_confidence': mean confidence,
       'true_attacks_caught': attacks with this action,
       'benign_affected': benign traffic with this action
   }
   ```

### 7. Save Mitigation Results
   **Create output folder:** `mitigation_data/{model_combination}/`
   
   **Save 5 files:**
   1. **mitigation_actions.csv** - Per-sample actions and outcomes
   2. **mitigation_metrics.csv** - Overall mitigation performance
   3. **mitigation_strategy_breakdown.csv** - Action type statistics
   4. **mitigation_config.json** - Thresholds and model configuration
   5. **combined_metrics.csv** - Ensemble detection + mitigation metrics

### 8. Print Summary Report
   Display formatted summary:
   - Overall performance (success rate, accuracy)
   - Mitigation effectiveness (precision, recall, F1)
   - Action breakdown (counts and percentages)
   - Mitigation outcomes (TP/FP/TN/FN)
   - Configuration (threshold values)

### 9. Return Mitigation Results
   ```python
   return {
       'mitigation_actions': actions_dataframe,
       'mitigation_metrics': performance_metrics,
       'strategy_breakdown': action_statistics,
       'ensemble_info': original_detection_results,
       'original_ensemble_results': complete_ensemble_output
   }
   ```

---

## Performance Metrics Calculated

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Threat detection accuracy |
| **Recall/Sensitivity** | TP / (TP + FN) | Attack detection rate |
| **Specificity** | TN / (TN + FP) | Benign traffic preservation |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean balance |
| **ROC-AUC** | Area under ROC curve | Threshold-independent performance |
| **PR-AUC** | Area under Precision-Recall curve | For imbalanced datasets |
| **Cohen's Kappa** | Agreement accounting for chance | Robust inter-rater agreement |

---

## Implementation Details

### Dataset Information
- **Source:** CIC-DDoS2019 Dataset
- **Original Size:** 85 columns, 225,745 samples
- **Final Features:** 38 features after preprocessing
- **Class Distribution:** Binary (0=BENIGN, 1=DDoS)
- **Processed Dataset:** `artifacts/ddos_processed_dataset.csv`

### Model Architecture Parameters
- **Window Size:** 10 time steps for LSTM/RNN models
- **Feature Count:** 38 input features
- **Batch Size:** 32-256 (configurable)
- **Device:** Auto-detect CUDA GPU or CPU

### Ensemble Configuration
- **Models:** 5 deep learning architectures
- **Weight Options:** Equal or custom based on validation performance
- **Normalization:** Weights automatically sum to 1.0
- **Fallback:** If model fails to load, continue with remaining models

### Performance Characteristics
- **Inference Speed:** ~10-50ms per sample on GPU
- **Memory:** 4GB+ RAM recommended
- **Batch Processing:** Efficient parallel computation
- **Output:** CSV files with comprehensive metrics

### File Paths
- **Models:** `models/*.pt` (5 pretrained models)
- **Scalers:** `scalers/` (fitted preprocessing scalers)
- **Ensemble Output:** `ensemble_data/{model_combination}/`
- **Mitigation Output:** `mitigation_data/{model_combination}/`

### Error Handling
- Multiple checkpoint loading approaches
- Fallback to alternative model versions (_ultra_clean, _minimal)
- Strict=False option for partial state dict loading
- Automatic weight renormalization if models fail
