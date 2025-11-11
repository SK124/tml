import torch
import numpy as np
import time
import json
import os
from datetime import datetime
from itertools import product
import pandas as pd
from part1 import * 



class BenchmarkConfig:
    
    # Fine-tuning strategies
    FINETUNE_TYPES = ['BASIC', 'PGD', 'FGSM']
    
    # Defense strategies
    DEFENSE_TYPES = [
        'none',
        'output_perturbation',
        'input_perturbation',
        'test_time_augmentation',
        'temperature_scaled',
        'response_limited_topk',
        'response_limited_hard',
        'adaptive_noise'
    ]
    
    # Hyperparameter ranges
    HYPERPARAMS = {
        'output_perturbation': {
            'scale': [0.05, 0.1, 0.2, 0.3, 0.5]
        },
        'input_perturbation': {
            'sigma': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        },
        'test_time_augmentation': {
            'num_augmentations': [1, 3, 5, 7, 10]
        },
        'temperature_scaled': {
            'temp': [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        },
        'response_limited_topk': {
            'top_k': [1, 2, 3, 4, 5]
        },
        'response_limited_hard': {
            'hard_label': [True]
        },
        'adaptive_noise': {
            'alpha': [0.1, 0.3, 0.5, 0.7, 1.0]
        },
        'none': {}
    }
    
    # Fine-tuning epochs
    FINETUNE_EPOCHS = [1, 3, 5, 7, 10]


class BenchmarkRunner:
    """Main benchmark runner"""
    
    def __init__(self, model_fp, data_dir='./data', output_dir='./benchmark_results', device='cuda'):
        self.model_fp = model_fp
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        
        print(f"Benchmark initialized - Device: {device}")
        print(f"Output directory: {output_dir}")
    
    def load_data(self):
        """Load all required datasets"""
        print("\n--- Loading datasets ---")
        
        self.train_loader = utils.make_loader(
            f'{self.data_dir}/train.npz', 'train_x', 'train_y', 
            batch_size=256, shuffle=False
        )
        
        self.val_loader = utils.make_loader(
            f'{self.data_dir}/valtest.npz', 'val_x', 'val_y', 
            batch_size=512, shuffle=False
        )
        
        # MIA data
        self.in_x, self.in_y = load_and_grab(
            f'{self.data_dir}/members.npz', 'members', num_batches=2
        )
        self.out_x, self.out_y = load_and_grab(
            f'{self.data_dir}/nonmembers.npz', 'nonmembers', num_batches=2
        )
        
        self.mia_eval_x = torch.cat([self.in_x, self.out_x], 0)
        self.mia_eval_y = torch.cat([
            torch.ones_like(self.in_y), 
            torch.zeros_like(self.out_y)
        ], 0).cpu().detach().numpy().reshape((-1, 1))
        
        # Load adversarial examples if available
        self.advex_data = None
        advexp_fp = 'advexp0.npz'
        if os.path.exists(advexp_fp):
            self.advex_data = load_advex(advexp_fp)
        
        print("Datasets loaded successfully")
    
    def get_predict_fn(self, model, defense_type, defense_params):
        """Get prediction function based on defense type"""
        if defense_type == 'none':
            return lambda x, dev: basic_predict(model, x, device=dev)
        elif defense_type == 'output_perturbation':
            return lambda x, dev: output_perturbation_predict(
                model, x, device=dev, **defense_params
            )
        elif defense_type == 'input_perturbation':
            return lambda x, dev: input_perturbation_predict(
                model, x, device=dev, **defense_params
            )
        elif defense_type == 'test_time_augmentation':
            return lambda x, dev: test_time_augmentation_predict(
                model, x, device=dev, **defense_params
            )
        elif defense_type == 'temperature_scaled':
            return lambda x, dev: temperature_scaled_predict(
                model, x, device=dev, **defense_params
            )
        elif defense_type == 'response_limited_topk':
            return lambda x, dev: response_limited_predict(
                model, x, device=dev, **defense_params
            )
        elif defense_type == 'response_limited_hard':
            return lambda x, dev: response_limited_predict(
                model, x, device=dev, **defense_params
            )
        elif defense_type == 'adaptive_noise':
            return lambda x, dev: adaptive_noise_injection(
                model, x, device=dev, **defense_params
            )
        else:
            raise ValueError(f"Unknown defense type: {defense_type}")
    
    def evaluate_utility(self, predict_fn):
        """Evaluate model utility (accuracy)"""
        train_acc = utils.eval_wrapper(predict_fn, self.train_loader, device=self.device)
        val_acc = utils.eval_wrapper(predict_fn, self.val_loader, device=self.device)
        return train_acc, val_acc
    
    def evaluate_privacy(self, predict_fn):
        """Evaluate privacy against MIA attacks"""
        results = {}
        
        # Simple confidence threshold MIA
        try:
            in_out_preds = simple_conf_threshold_mia(
                predict_fn, self.mia_eval_x, device=self.device
            ).reshape((-1, 1))
            results['conf_threshold'] = self._compute_mia_metrics(in_out_preds)
        except Exception as e:
            results['conf_threshold'] = {'error': str(e)}
        
        # Simple logits threshold MIA
        try:
            in_out_preds = simple_logits_threshold_mia(
                predict_fn, self.mia_eval_x, device=self.device
            ).reshape((-1, 1))
            results['logits_threshold'] = self._compute_mia_metrics(in_out_preds)
        except Exception as e:
            results['logits_threshold'] = {'error': str(e)}
        
        # Modified entropy MIA
        try:
            in_out_preds = modified_entropy_mia(
                predict_fn, self.mia_eval_x, device=self.device
            ).reshape((-1, 1))
            results['modified_entropy'] = self._compute_mia_metrics(in_out_preds)
        except Exception as e:
            results['modified_entropy'] = {'error': str(e)}
        
        return results
    
    def _compute_mia_metrics(self, predictions):
        """Compute MIA metrics from predictions"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(self.mia_eval_y, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        tol = 1e-10
        attack_acc = np.trace(cm) / np.sum(np.sum(cm))
        attack_tpr = tp / (tp + fn + tol)
        attack_fpr = fp / (fp + tn + tol)
        attack_adv = attack_tpr - attack_fpr
        attack_precision = tp / (tp + fp + tol)
        attack_recall = tp / (tp + fn + tol)
        attack_f1 = tp / (tp + 0.5 * (fp + fn) + tol)
        
        return {
            'accuracy': float(attack_acc),
            'tpr': float(attack_tpr),
            'fpr': float(attack_fpr),
            'advantage': float(attack_adv),
            'precision': float(attack_precision),
            'recall': float(attack_recall),
            'f1': float(attack_f1)
        }
    
    def evaluate_robustness(self, predict_fn):
        """Evaluate robustness against adversarial examples"""
        if self.advex_data is None:
            return None
        
        adv_x, benign_x, benign_y = self.advex_data
        benign_y = benign_y.flatten()
        
        benign_pred_y = predict_fn(torch.from_numpy(benign_x), self.device).cpu().numpy()
        benign_pred_y = np.argmax(benign_pred_y, axis=-1).astype(int)
        benign_acc = np.mean(benign_y == benign_pred_y)
        
        adv_pred_y = predict_fn(torch.from_numpy(adv_x), self.device).cpu().numpy()
        adv_pred_y = np.argmax(adv_pred_y, axis=-1).astype(int)
        adv_acc = np.mean(benign_y == adv_pred_y)
        
        return {
            'benign_accuracy': float(benign_acc),
            'adversarial_accuracy': float(adv_acc),
            'robustness_drop': float(benign_acc - adv_acc)
        }
    
    def run_single_experiment(self, finetune_type, num_epochs, defense_type, defense_params):
        """Run a single experiment with given configuration"""
        
        exp_id = f"{finetune_type}_ep{num_epochs}_{defense_type}_" + \
                 "_".join([f"{k}{v}" for k, v in defense_params.items()])
        
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_id}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Load and fine-tune model
            model, _ = utils.load_model(self.model_fp, device=self.device)
            
            print(f"Fine-tuning with {finetune_type} for {num_epochs} epochs...")
            finetune_type_enum = getattr(FineTuneType, finetune_type)
            model = fine_tune(
                model, self.train_loader, 
                type=finetune_type_enum, 
                device=self.device, 
                num_epochs=num_epochs
            )
            
            # Get prediction function
            predict_fn = self.get_predict_fn(model, defense_type, defense_params)
            
            # Evaluate utility
            print("Evaluating utility...")
            train_acc, val_acc = self.evaluate_utility(predict_fn)
            
            # Evaluate privacy
            print("Evaluating privacy...")
            privacy_results = self.evaluate_privacy(predict_fn)
            
            # Evaluate robustness
            print("Evaluating robustness...")
            robustness_results = self.evaluate_robustness(predict_fn)
            
            elapsed_time = time.time() - start_time
            
            # Compile results
            result = {
                'experiment_id': exp_id,
                'timestamp': datetime.now().isoformat(),
                'finetune_type': finetune_type,
                'num_epochs': num_epochs,
                'defense_type': defense_type,
                'defense_params': defense_params,
                'train_accuracy': float(train_acc),
                'val_accuracy': float(val_acc),
                'privacy': privacy_results,
                'robustness': robustness_results,
                'elapsed_time': elapsed_time,
                'status': 'success'
            }
            
            print(f"\nResults:")
            print(f"  Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Time: {elapsed_time:.2f}s")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            result = {
                'experiment_id': exp_id,
                'timestamp': datetime.now().isoformat(),
                'finetune_type': finetune_type,
                'num_epochs': num_epochs,
                'defense_type': defense_type,
                'defense_params': defense_params,
                'status': 'failed',
                'error': str(e)
            }
        
        # Save individual result
        self.save_result(result)
        self.results.append(result)
        
        return result
    
    def save_result(self, result):
        """Save individual result to file"""
        filename = f"{result['experiment_id']}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    def run_full_benchmark(self, quick_mode=False):
        """Run full benchmark across all configurations"""
        
        print("\n" + "="*60)
        print("STARTING FULL BENCHMARK")
        print("="*60)
        
        total_experiments = 0
        
        # Count total experiments
        for ft_type in BenchmarkConfig.FINETUNE_TYPES:
            epochs_to_test = [1, 5] if quick_mode else BenchmarkConfig.FINETUNE_EPOCHS
            for epochs in epochs_to_test:
                for def_type in BenchmarkConfig.DEFENSE_TYPES:
                    params_dict = BenchmarkConfig.HYPERPARAMS[def_type]
                    
                    if not params_dict:
                        total_experiments += 1
                    else:
                        # Generate all combinations of hyperparameters
                        param_names = list(params_dict.keys())
                        param_values = list(params_dict.values())
                        
                        if quick_mode:
                            # Take only first and last value for each param
                            param_values = [[v[0], v[-1]] if len(v) > 1 else v 
                                          for v in param_values]
                        
                        for combo in product(*param_values):
                            total_experiments += 1
        
        print(f"Total experiments to run: {total_experiments}")
        
        # Run experiments
        experiment_num = 0
        
        for ft_type in BenchmarkConfig.FINETUNE_TYPES:
            epochs_to_test = [1, 5] if quick_mode else BenchmarkConfig.FINETUNE_EPOCHS
            
            for epochs in epochs_to_test:
                for def_type in BenchmarkConfig.DEFENSE_TYPES:
                    params_dict = BenchmarkConfig.HYPERPARAMS[def_type]
                    
                    if not params_dict:
                        # No hyperparameters for this defense
                        experiment_num += 1
                        print(f"\n[{experiment_num}/{total_experiments}]")
                        self.run_single_experiment(ft_type, epochs, def_type, {})
                    else:
                        # Generate all combinations of hyperparameters
                        param_names = list(params_dict.keys())
                        param_values = list(params_dict.values())
                        
                        if quick_mode:
                            param_values = [[v[0], v[-1]] if len(v) > 1 else v 
                                          for v in param_values]
                        
                        for combo in product(*param_values):
                            params = dict(zip(param_names, combo))
                            experiment_num += 1
                            print(f"\n[{experiment_num}/{total_experiments}]")
                            self.run_single_experiment(ft_type, epochs, def_type, params)
        
        # Generate summary
        self.generate_summary()
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)
    
    def generate_summary(self):
        """Generate summary report of all experiments"""
        
        summary_file = os.path.join(self.output_dir, 'summary.csv')
        
        # Convert results to DataFrame
        rows = []
        for r in self.results:
            if r['status'] == 'success':
                row = {
                    'experiment_id': r['experiment_id'],
                    'finetune_type': r['finetune_type'],
                    'num_epochs': r['num_epochs'],
                    'defense_type': r['defense_type'],
                    'train_acc': r['train_accuracy'],
                    'val_acc': r['val_accuracy'],
                    'elapsed_time': r['elapsed_time']
                }
                
                # Add defense params
                for k, v in r['defense_params'].items():
                    row[f'param_{k}'] = v
                
                # Add privacy metrics (average across attacks)
                privacy_accs = []
                for attack_name, metrics in r['privacy'].items():
                    if 'error' not in metrics:
                        privacy_accs.append(metrics['accuracy'])
                        row[f'mia_{attack_name}_acc'] = metrics['accuracy']
                        row[f'mia_{attack_name}_adv'] = metrics['advantage']
                
                if privacy_accs:
                    row['mia_avg_acc'] = np.mean(privacy_accs)
                
                # Add robustness metrics
                if r['robustness']:
                    row['adv_benign_acc'] = r['robustness']['benign_accuracy']
                    row['adv_robust_acc'] = r['robustness']['adversarial_accuracy']
                    row['adv_drop'] = r['robustness']['robustness_drop']
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(summary_file, index=False)
        
        print(f"\nSummary saved to: {summary_file}")
        
        # Print top performers
        if len(df) > 0:
            print("\n--- Top 5 by Validation Accuracy ---")
            print(df.nlargest(5, 'val_acc')[['experiment_id', 'val_acc', 'train_acc']])
            
            if 'mia_avg_acc' in df.columns:
                print("\n--- Top 5 by Privacy (Lowest MIA Accuracy) ---")
                print(df.nsmallest(5, 'mia_avg_acc')[['experiment_id', 'val_acc', 'mia_avg_acc']])
            
            if 'adv_robust_acc' in df.columns:
                print("\n--- Top 5 by Robustness (Highest Adversarial Accuracy) ---")
                print(df.nlargest(5, 'adv_robust_acc')[['experiment_id', 'val_acc', 'adv_robust_acc']])


def main():
    """Main execution function"""
    
    # Configuration
    MODEL_FP = './target_model.pt'
    DATA_DIR = './data'
    OUTPUT_DIR = './benchmark_results'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set to True for faster testing (uses fewer hyperparameter values)
    QUICK_MODE = False
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(
        model_fp=MODEL_FP,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )
    
    # Load data
    runner.load_data()
    
    # Run benchmark
    runner.run_full_benchmark(quick_mode=QUICK_MODE)


if __name__ == "__main__":
    main()