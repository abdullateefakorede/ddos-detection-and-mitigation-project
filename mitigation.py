from datetime import datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd


class DDoSMitigator:
    """
    DDoS Mitigation Module that works with ModelEnsemble predictions.

    Applies mitigation strategies based on ensemble detection probabilities
    and tracks comprehensive mitigation metrics.
    """

    def __init__(self,
                 high_confidence_threshold=0.9,
                 medium_confidence_threshold=0.7):
        """
        Initialize mitigator with confidence thresholds.

        Args:
            high_confidence_threshold: Probability threshold for immediate blocking
            medium_confidence_threshold: Probability threshold for rate limiting
            low_confidence_threshold: Probability threshold for monitoring
        """
        self.thresholds = {
            'high': high_confidence_threshold,
            'medium': medium_confidence_threshold,
        }
        self.mitigation_history = []

    def mitigate(self, ensemble_results):
        """
        Main method: Apply mitigation based on ensemble predictions.

        Args:
            ensemble_results: Dictionary returned from ModelEnsemble.predict()
                {
                    'ensemble_probabilities': array of probabilities,
                    'ensemble_predictions': array of binary predictions,
                    'true_labels': array of true labels,
                    'individual_predictions': dict of individual model predictions,
                    'individual_probabilities': dict of individual model probabilities,
                    'ensemble_metrics': dict of ensemble performance metrics,
                    'individual_metrics': dict of individual model metrics,
                    'model_names': list of model names,
                    'weights': list of model weights
                }

        Returns:
            Dictionary containing:
                - mitigation_actions: DataFrame with per-sample actions
                - mitigation_metrics: Dictionary of overall metrics
                - strategy_breakdown: Dictionary of strategy statistics
                - ensemble_info: Original ensemble information
        """
        # Extract key components
        probabilities = ensemble_results['ensemble_probabilities']
        predictions = ensemble_results['ensemble_predictions']
        true_labels = ensemble_results['true_labels']

        # Apply mitigation strategies
        mitigation_actions = self._apply_mitigation_strategies(
            probabilities, predictions, true_labels
        )

        # Calculate mitigation performance metrics
        mitigation_metrics = self._calculate_mitigation_metrics(
            mitigation_actions, true_labels
        )

        # Get strategy breakdown
        strategy_breakdown = self._get_strategy_breakdown(mitigation_actions)

        # Store in history
        self.mitigation_history.append({
            'timestamp': datetime.now(),
            'num_samples': len(probabilities),
            'metrics': mitigation_metrics,
            'strategy_breakdown': strategy_breakdown
        })

        return {
            'mitigation_actions': mitigation_actions,
            'mitigation_metrics': mitigation_metrics,
            'strategy_breakdown': strategy_breakdown,
            'ensemble_info': {
                'model_names': ensemble_results['model_names'],
                'weights': ensemble_results['weights'],
                'ensemble_metrics': ensemble_results['ensemble_metrics'],
                'individual_metrics': ensemble_results['individual_metrics']
            },
            'original_ensemble_results': ensemble_results
        }

    def _apply_mitigation_strategies(self, probabilities, predictions, true_labels):
        """
        Apply mitigation strategy to each sample based on probability.

        Returns:
            DataFrame with columns: sample_id, probability, prediction,
                                   true_label, action, confidence_level,
                                   is_correct_action
        """
        actions = []

        for idx, (prob, pred, true_label) in enumerate(zip(probabilities, predictions, true_labels)):
            # Determine action based on probability thresholds
            if prob >= self.thresholds['high']:
                action = 'BLOCK'
                confidence_level = 'HIGH'
            elif prob >= self.thresholds['medium']:
                action = 'RATE_LIMIT'
                confidence_level = 'MEDIUM'
            else:
                action = 'ALLOW'
                confidence_level = 'VERY_LOW'

            # Determine if action was correct
            # Correct if: (predicted attack AND true attack) OR (predicted benign AND true benign)
            is_correct_action = self._is_action_correct(
                action, true_label)

            # Determine action outcome
            if true_label == 1:  # Actual attack
                if action in ['BLOCK', 'RATE_LIMIT']:
                    outcome = 'TRUE_POSITIVE_MITIGATION'  # Successfully mitigated
                elif action == 'MONITOR':
                    outcome = 'PARTIAL_MITIGATION'  # Detected but not fully mitigated
                else:
                    outcome = 'FALSE_NEGATIVE_MITIGATION'  # Missed attack
            else:  # Actual benign
                if action in ['BLOCK', 'RATE_LIMIT']:
                    outcome = 'FALSE_POSITIVE_MITIGATION'  # Incorrectly blocked legitimate traffic
                else:
                    outcome = 'TRUE_NEGATIVE_MITIGATION'  # Correctly allowed

            actions.append({
                'sample_id': idx,
                'probability': prob,
                'prediction': pred,
                'true_label': true_label,
                'action': action,
                'confidence_level': confidence_level,
                'is_correct_action': is_correct_action,
                'outcome': outcome
            })

        return pd.DataFrame(actions)

    def _is_action_correct(self, action, true_label):
        """
        Determine if mitigation action was appropriate.

        Correct actions:
        - BLOCK/RATE_LIMIT for actual attacks (true_label=1)
        - ALLOW/MONITOR for benign traffic (true_label=0)
        """
        if true_label == 1:  # Actual attack
            return action in ['BLOCK', 'RATE_LIMIT', 'MONITOR']
        else:  # Benign traffic
            return action in ['ALLOW', 'MONITOR']

    def _calculate_mitigation_metrics(self, actions_df, true_labels):
        """
        Calculate comprehensive mitigation performance metrics.

        Similar to detection metrics but focused on mitigation effectiveness.
        """
        # Basic counts
        total_samples = len(actions_df)
        actual_attacks = np.sum(true_labels == 1)
        actual_benign = np.sum(true_labels == 0)

        # Mitigation outcome counts
        tp_mitigation = len(
            actions_df[actions_df['outcome'] == 'TRUE_POSITIVE_MITIGATION'])
        fp_mitigation = len(
            actions_df[actions_df['outcome'] == 'FALSE_POSITIVE_MITIGATION'])
        tn_mitigation = len(
            actions_df[actions_df['outcome'] == 'TRUE_NEGATIVE_MITIGATION'])
        fn_mitigation = len(
            actions_df[actions_df['outcome'] == 'FALSE_NEGATIVE_MITIGATION'])
        partial_mitigation = len(
            actions_df[actions_df['outcome'] == 'PARTIAL_MITIGATION'])

        # Core mitigation metrics
        mitigation_success_rate = tp_mitigation / \
            actual_attacks if actual_attacks > 0 else 0
        false_positive_mitigation_rate = fp_mitigation / \
            actual_benign if actual_benign > 0 else 0
        legitimate_traffic_preservation = tn_mitigation / \
            actual_benign if actual_benign > 0 else 0

        # Action-based metrics
        blocked = len(actions_df[actions_df['action'] == 'BLOCK'])
        rate_limited = len(actions_df[actions_df['action'] == 'RATE_LIMIT'])
        monitored = len(actions_df[actions_df['action'] == 'MONITOR'])
        allowed = len(actions_df[actions_df['action'] == 'ALLOW'])

        # Precision and recall for mitigation
        precision = tp_mitigation / \
            (tp_mitigation + fp_mitigation) if (tp_mitigation + fp_mitigation) > 0 else 0
        recall = tp_mitigation / \
            (tp_mitigation + fn_mitigation) if (tp_mitigation + fn_mitigation) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision +
                                               recall) if (precision + recall) > 0 else 0

        # Overall accuracy
        correct_actions = actions_df['is_correct_action'].sum()
        mitigation_accuracy = correct_actions / total_samples

        # Average confidence for different outcomes
        avg_confidence_tp = actions_df[actions_df['outcome']
                                       == 'TRUE_POSITIVE_MITIGATION']['probability'].mean()
        avg_confidence_fp = actions_df[actions_df['outcome']
                                       == 'FALSE_POSITIVE_MITIGATION']['probability'].mean()

        # Traffic impact metrics
        total_mitigated = blocked + rate_limited
        mitigation_impact = total_mitigated / total_samples

        return {
            # Sample counts
            'total_samples': total_samples,
            'actual_attacks': int(actual_attacks),
            'actual_benign': int(actual_benign),

            # Mitigation outcomes
            'true_positive_mitigation': tp_mitigation,
            'false_positive_mitigation': fp_mitigation,
            'true_negative_mitigation': tn_mitigation,
            'false_negative_mitigation': fn_mitigation,
            'partial_mitigation': partial_mitigation,

            # Core performance metrics
            'mitigation_success_rate': float(mitigation_success_rate),
            'false_positive_mitigation_rate': float(false_positive_mitigation_rate),
            'legitimate_traffic_preservation': float(legitimate_traffic_preservation),
            'mitigation_accuracy': float(mitigation_accuracy),

            # Precision, Recall, F1
            'mitigation_precision': float(precision),
            'mitigation_recall': float(recall),
            'mitigation_f1_score': float(f1_score),

            # Action counts
            'blocked_count': blocked,
            'rate_limited_count': rate_limited,
            'monitored_count': monitored,
            'allowed_count': allowed,

            # Confidence metrics
            'avg_confidence_true_positive': float(avg_confidence_tp) if not np.isnan(avg_confidence_tp) else 0.0,
            'avg_confidence_false_positive': float(avg_confidence_fp) if not np.isnan(avg_confidence_fp) else 0.0,

            # Impact metrics
            'total_mitigated': total_mitigated,
            'mitigation_impact_percentage': float(mitigation_impact * 100),

            # Thresholds used
            'high_threshold': self.thresholds['high'],
            'medium_threshold': self.thresholds['medium'],
        }

    def _get_strategy_breakdown(self, actions_df):
        """
        Get detailed breakdown of mitigation strategies and their effectiveness.
        """
        breakdown = {}

        for action in ['BLOCK', 'RATE_LIMIT', 'MONITOR', 'ALLOW']:
            action_df = actions_df[actions_df['action'] == action]

            if len(action_df) > 0:
                breakdown[action] = {
                    'count': len(action_df),
                    'percentage': len(action_df) / len(actions_df) * 100,
                    'correct_actions': action_df['is_correct_action'].sum(),
                    'accuracy': action_df['is_correct_action'].sum() / len(action_df),
                    'avg_probability': float(action_df['probability'].mean()),
                    'true_attacks_caught': len(action_df[action_df['true_label'] == 1]),
                    'benign_affected': len(action_df[action_df['true_label'] == 0])
                }
            else:
                breakdown[action] = {
                    'count': 0,
                    'percentage': 0.0,
                    'correct_actions': 0,
                    'accuracy': 0.0,
                    'avg_probability': 0.0,
                    'true_attacks_caught': 0,
                    'benign_affected': 0
                }

        return breakdown

    def save_results(self, mitigation_results, output_dir, model_config_name=None):
        """
        Save mitigation results to organized directory structure.

        Mirrors the detection output structure for consistency.

        Args:
            mitigation_results: Dictionary returned from mitigate()
            output_dir: Base directory for mitigation outputs
            model_config_name: Optional name (e.g., "DBNNetwork-0.333_DDoSTransformer-0.333")
        """
        output_path = Path(output_dir)

        # Create subdirectory for this model configuration if provided
        if model_config_name:
            output_path = output_path / model_config_name

        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Save mitigation actions (per-sample details)
        actions_df = mitigation_results['mitigation_actions']
        actions_df.to_csv(output_path / 'mitigation_actions.csv', index=False)

        # 2. Save overall metrics
        metrics_df = pd.DataFrame([mitigation_results['mitigation_metrics']])
        metrics_df.to_csv(output_path / 'mitigation_metrics.csv', index=False)

        # 3. Save strategy breakdown
        breakdown_df = pd.DataFrame(
            mitigation_results['strategy_breakdown']).T
        breakdown_df.to_csv(output_path / 'mitigation_strategy_breakdown.csv')

        # 4.  Save configuration
        config = {
            'thresholds': self.thresholds,
            'model_names': mitigation_results['ensemble_info']['model_names'],
            'model_weights': mitigation_results['ensemble_info']['weights'],
            'timestamp': datetime.now().isoformat()
        }
        with open(output_path / 'mitigation_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # 5. Save combined ensemble + mitigation metrics
        combined_metrics = {
            **mitigation_results['ensemble_info']['ensemble_metrics'],
            **mitigation_results['mitigation_metrics']
        }
        combined_df = pd.DataFrame([combined_metrics])
        combined_df.to_csv(
            output_path / 'combined_detection_mitigation_metrics.csv', index=False)

        print(f"✅ Mitigation results saved to: {output_path}")
        print(f"- mitigation_actions.csv: {len(actions_df)} samples")
        print("- mitigation_metrics.csv: Overall performance metrics")
        print("- mitigation_strategy_breakdown.csv: Per-strategy analysis")
        print("- mitigation_config.json: Configuration and thresholds")
        print("- combined_detection_mitigation_metrics.csv: Full pipeline metrics")
        return output_path

    def print_summary(self, mitigation_results):
        """
        Print a formatted summary of mitigation results.
        """
        metrics = mitigation_results['mitigation_metrics']
        breakdown = mitigation_results['strategy_breakdown']

        print("\n" + "="*70)
        print("🛡️  DDoS MITIGATION SUMMARY")
        print("="*70)

        print("\n📊 Overall Performance:")
        print(f"   Total Samples: {metrics['total_samples']}")
        print(f"   Actual Attacks: {metrics['actual_attacks']}")
        print(f"   Actual Benign: {metrics['actual_benign']}")
        print(
            f"   Mitigation Success Rate: {metrics['mitigation_success_rate']:.2%}")
        print(f"   Mitigation Accuracy: {metrics['mitigation_accuracy']:.2%}")
        print(
            f"   Legitimate Traffic Preservation: {metrics['legitimate_traffic_preservation']:.2%}")

        print("\n🎯 Mitigation Effectiveness:")
        print(f"   Precision: {metrics['mitigation_precision']:.4f}")
        print(f"   Recall: {metrics['mitigation_recall']:.4f}")
        print(f"   F1 Score: {metrics['mitigation_f1_score']:.4f}")

        print("\n⚡ Actions Taken:")
        print(
            f"   BLOCKED: {breakdown['BLOCK']['count']} ({breakdown['BLOCK']['percentage']:.1f}%)")
        print(
            f"   RATE LIMITED: {breakdown['RATE_LIMIT']['count']} ({breakdown['RATE_LIMIT']['percentage']:.1f}%)")
        print(
            f"   MONITORED: {breakdown['MONITOR']['count']} ({breakdown['MONITOR']['percentage']:.1f}%)")
        print(
            f"   ALLOWED: {breakdown['ALLOW']['count']} ({breakdown['ALLOW']['percentage']:.1f}%)")

        print("\n🔍 Mitigation Outcomes:")
        print(
            f"   True Positive Mitigation: {metrics['true_positive_mitigation']}")
        print(
            f"   False Positive Mitigation: {metrics['false_positive_mitigation']}")
        print(
            f"   True Negative Mitigation: {metrics['true_negative_mitigation']}")
        print(
            f"   False Negative Mitigation: {metrics['false_negative_mitigation']}")

        print("\n⚙️ Configuration:")
        print(f"High Confidence Threshold: {metrics['high_threshold']}")
        print(f"Medium Confidence Threshold: {metrics['medium_threshold']}")

        print("="*70 + "\n")
