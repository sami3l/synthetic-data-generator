# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# SDV imports
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

class SyntheticDataGenerator:
    def __init__(self):
        self.real_data = None
        self.synthetic_data = None
        self.synthesizer = None
        self.metadata = None
        self.quality_report = {}
        
    def welcome_message(self):
        """Display welcome message"""
        print("="*60)
        print("SDV GAN Synthetic Data Generator")
        print("="*60)
        print("This program generates synthetic data using CTGAN")
        print("Please follow the prompts to configure your settings")
        print("="*60)
    
    def get_data_parameters(self):
        """Get data input parameters from user"""
        print("\n DATA PARAMETERS")
        print("-" * 30)
        
        # Get input file path
        while True:
            file_path = input("Enter the path to your CSV file: ").strip()
            if Path(file_path).exists() and file_path.endswith('.csv'):
                self.data_path = file_path
                break
            else:
                print("File not found or not a CSV file. Please try again.")
        
        # Missing value handling
        print("\nMissing Value Handling Options:")
        print("1. Drop rows with missing values")
        print("2. Fill with mean/mode")
        print("3. Keep as is (CTGAN can handle some missing values)")
        
        while True:
            choice = input("Select option (1-3): ").strip()
            if choice in ['1', '2', '3']:
                self.missing_value_strategy = choice
                break
            else:
                print("Please enter 1, 2, or 3")
    
    def get_model_parameters(self):
        """Get CTGAN model parameters from user"""
        print("\n MODEL PARAMETERS")
        print("-" * 30)
        
        # Number of epochs
        while True:
            try:
                epochs = int(input("Number of training epochs (default 300): ") or "300")
                if epochs > 0:
                    self.epochs = epochs
                    break
                else:
                    print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")
        
        # Batch size
        while True:
            try:
                batch_size = int(input("Batch size (default 500): ") or "500")
                if batch_size > 0:
                    self.batch_size = batch_size
                    break
                else:
                    print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")
        
        # Learning rate
        while True:
            try:
                lr = float(input("Learning rate (default 0.0002): ") or "0.0002")
                if lr > 0:
                    self.learning_rate = lr
                    break
                else:
                    print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")
        
        # Generator/Discriminator dimensions
        while True:
            try:
                gen_dim = input("Generator dimensions (default 256,256): ") or "256,256"
                gen_dim = tuple(map(int, gen_dim.split(',')))
                self.generator_dim = gen_dim
                break
            except ValueError:
                print("Please enter dimensions like: 256,256")
        
        while True:
            try:
                disc_dim = input("Discriminator dimensions (default 256,256): ") or "256,256"
                disc_dim = tuple(map(int, disc_dim.split(',')))
                self.discriminator_dim = disc_dim
                break
            except ValueError:
                print("Please enter dimensions like: 256,256")
        
        # Number of synthetic samples
        while True:
            try:
                num_samples = int(input("Number of synthetic samples to generate: "))
                if num_samples > 0:
                    self.num_samples = num_samples
                    break
                else:
                    print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")
    
    def get_output_parameters(self):
        """Get output parameters from user"""
        print("\nOUTPUT PARAMETERS")
        print("-" * 30)
        
        # Output format
        while True:
            format_choice = input("Output format (CSV/JSON): ").strip().upper()
            if format_choice in ['CSV', 'JSON']:
                self.output_format = format_choice
                break
            else:
                print("Please enter CSV or JSON")
        
        # Output location
        while True:
            output_path = input("Output directory (default: current directory): ").strip() or "."
            if Path(output_path).exists():
                self.output_path = output_path
                break
            else:
                print("Directory doesn't exist. Please try again.")
        
        # Quality report
        quality_report = input("Generate quality report with visualizations? (y/n): ").strip().lower()
        self.generate_quality_report = quality_report in ['y', 'yes']
    
    def load_and_validate_data(self):
        """Load and validate the input data"""
        print("\nLOADING DATA")
        print("-" * 30)
        
        try:
            # Load data
            self.real_data = pd.read_csv(self.data_path)
            print(f" Data loaded successfully!")
            print(f"   Shape: {self.real_data.shape}")
            print(f"   Columns: {list(self.real_data.columns)}")
            
            # Handle missing values
            if self.missing_value_strategy == '1':
                self.real_data = self.real_data.dropna()
                print(" Dropped rows with missing values")
            elif self.missing_value_strategy == '2':
                # Fill numeric columns with mean, categorical with mode
                for col in self.real_data.columns:
                    if self.real_data[col].dtype in ['int64', 'float64']:
                        self.real_data[col].fillna(self.real_data[col].mean(), inplace=True)
                    else:
                        self.real_data[col].fillna(self.real_data[col].mode()[0], inplace=True)
                print(" Filled missing values ")
            else:
                print(" Keeping missing values as is")
            
            # Create metadata
            self.metadata = SingleTableMetadata()
            self.metadata.detect_from_dataframe(self.real_data)
            print(" Metadata detected automatically")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        
        return True
    
    def train_model(self):
        """Train the CTGAN model"""
        print("\nTRAINING MODEL")
        print("-" * 30)
        
        try:
            # Initialize CTGAN
            self.synthesizer = CTGANSynthesizer(
                metadata=self.metadata,
                epochs=self.epochs,
                batch_size=self.batch_size,
                generator_lr=self.learning_rate,
                discriminator_lr=self.learning_rate,
                generator_dim=self.generator_dim,
                discriminator_dim=self.discriminator_dim,
                verbose=True
            )
            
            print("Training CTGAN model...")
            print(f"   Epochs: {self.epochs}")
            print(f"   Batch size: {self.batch_size}")
            print(f"   Learning rate: {self.learning_rate}")
            
            # Train the model
            self.synthesizer.fit(self.real_data)
            print(" Model trained successfully!")
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
        
        return True
    
    def generate_synthetic_data(self):
        """Generate synthetic data"""
        print("\nGENERATING SYNTHETIC DATA")
        print("-" * 30)
        
        try:
            print(f"Generating {self.num_samples} synthetic samples...")
            self.synthetic_data = self.synthesizer.sample(num_rows=self.num_samples)
            print(" Synthetic data generated successfully!")
            print(f"   Shape: {self.synthetic_data.shape}")
            
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return False
        
        return True
    
    def save_data(self):
        """Save synthetic data to file"""
        print("\nSAVING DATA")
        print("-" * 30)
        
        try:
            # Create filename
            base_name = Path(self.data_path).stem
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            if self.output_format == 'CSV':
                filename = f"{base_name}_synthetic_{timestamp}.csv"
                filepath = Path(self.output_path) / filename
                self.synthetic_data.to_csv(filepath, index=False)
            else: # JSON format
                filename = f"{base_name}_synthetic_{timestamp}.json"
                filepath = Path(self.output_path) / filename
                self.synthetic_data.to_json(filepath, orient='records', indent=2)
            
            print(f" Synthetic data saved to: {filepath}")
            self.output_file = filepath
            
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
        
        return True
    
    def calculate_statistical_metrics(self):
        """Calculate statistical quality metrics"""
        print("\nCalculating statistical metrics...")
        
        metrics = {}
        
        # Separate numeric and categorical columns
        numeric_cols = self.real_data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.real_data.select_dtypes(exclude=[np.number]).columns
        
        # Numeric column analysis
        for col in numeric_cols:
            real_values = self.real_data[col].dropna()
            synthetic_values = self.synthetic_data[col].dropna()
            
            # Basic statistics comparison
            real_stats = {
                'mean': real_values.mean(),
                'std': real_values.std(),
                'min': real_values.min(),
                'max': real_values.max(),
                'median': real_values.median(),
                'skewness': stats.skew(real_values),
                'kurtosis': stats.kurtosis(real_values)
            }
            
            synthetic_stats = {
                'mean': synthetic_values.mean(),
                'std': synthetic_values.std(),
                'min': synthetic_values.min(),
                'max': synthetic_values.max(),
                'median': synthetic_values.median(),
                'skewness': stats.skew(synthetic_values),
                'kurtosis': stats.kurtosis(synthetic_values)
            }
            
            # Statistical tests
            ks_stat, ks_p_value = stats.ks_2samp(real_values, synthetic_values)
            
            metrics[col] = {
                'type': 'numeric',
                'real_stats': real_stats,
                'synthetic_stats': synthetic_stats,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'distribution_similarity': 1 - ks_stat  # Higher is better
            }
        
        # Categorical column analysis
        for col in categorical_cols:
            real_counts = self.real_data[col].value_counts(normalize=True)
            synthetic_counts = self.synthetic_data[col].value_counts(normalize=True)
            
            # Align indices
            all_categories = set(real_counts.index) | set(synthetic_counts.index)
            real_aligned = real_counts.reindex(all_categories, fill_value=0)
            synthetic_aligned = synthetic_counts.reindex(all_categories, fill_value=0)
            
            # Chi-square test
            chi2_stat, chi2_p_value = stats.chisquare(synthetic_aligned, real_aligned)
            
            # Total variation distance
            tvd = 0.5 * np.sum(np.abs(real_aligned - synthetic_aligned))
            
            metrics[col] = {
                'type': 'categorical',
                'real_distribution': real_counts.to_dict(),
                'synthetic_distribution': synthetic_counts.to_dict(),
                'chi2_statistic': chi2_stat,
                'chi2_p_value': chi2_p_value,
                'total_variation_distance': tvd,
                'distribution_similarity': 1 - tvd  # Higher is better
            }
        
        return metrics
    
    def calculate_correlation_analysis(self):
        """Analyze correlation preservation"""
        print("Calculating correlation analysis...")
        
        numeric_cols = self.real_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'error': 'Not enough numeric columns for correlation analysis'}
        
        real_corr = self.real_data[numeric_cols].corr()
        synthetic_corr = self.synthetic_data[numeric_cols].corr()
        
        # Correlation difference
        corr_diff = np.abs(real_corr - synthetic_corr)
        mean_corr_diff = corr_diff.mean().mean()
        
        # Frobenius norm of correlation difference
        frobenius_norm = np.linalg.norm(real_corr - synthetic_corr, 'fro')
        
        return {
            'real_correlation': real_corr.to_dict(),
            'synthetic_correlation': synthetic_corr.to_dict(),
            'mean_correlation_difference': mean_corr_diff,
            'frobenius_norm': frobenius_norm,
            'correlation_preservation_score': 1 / (1 + mean_corr_diff)  # Higher is better
        }
    
    def calculate_machine_learning_efficacy(self):
        """Test ML model performance on real vs synthetic data"""
        print("Calculating ML efficacy...")
        
        try:
            # Use the first column as target (or specify a target column)
            target_col = self.real_data.columns[0]
            feature_cols = [col for col in self.real_data.columns if col != target_col]
            
            # Prepare data for classification/regression
            X_real = self.real_data[feature_cols]
            y_real = self.real_data[target_col]
            
            X_synthetic = self.synthetic_data[feature_cols]
            y_synthetic = self.synthetic_data[target_col]
            
            # Handle categorical variables (simple label encoding)
            for col in feature_cols:
                if X_real[col].dtype == 'object':
                    unique_vals = list(set(X_real[col].unique()) | set(X_synthetic[col].unique()))
                    val_map = {val: i for i, val in enumerate(unique_vals)}
                    X_real[col] = X_real[col].map(val_map)
                    X_synthetic[col] = X_synthetic[col].map(val_map)
            
            # Train-test split
            X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                X_real, y_real, test_size=0.2, random_state=42
            )
            
            # Train model on real data
            rf_real = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_real.fit(X_train_real, y_train_real)
            real_score = rf_real.score(X_test_real, y_test_real)
            
            # Train model on synthetic data
            rf_synthetic = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_synthetic.fit(X_synthetic, y_synthetic)
            synthetic_score = rf_synthetic.score(X_test_real, y_test_real)
            
            return {
                'real_data_performance': real_score,
                'synthetic_data_performance': synthetic_score,
                'performance_ratio': synthetic_score / real_score if real_score > 0 else 0,
                'target_column': target_col
            }
            
        except Exception as e:
            return {'error': f'ML efficacy calculation failed: {str(e)}'}
    
    def generate_visualizations(self):
        """Generate quality assessment visualizations"""
        print("Generating visualizations...")
        
        base_name = Path(self.data_path).stem
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory for plots
        plots_dir = Path(self.output_path) / f"{base_name}_quality_plots_{timestamp}"
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Distribution comparison plots
        numeric_cols = self.real_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:4]):  # Show first 4 numeric columns
                if i < len(axes):
                    axes[i].hist(self.real_data[col].dropna(), alpha=0.7, label='Real', bins=30, density=True)
                    axes[i].hist(self.synthetic_data[col].dropna(), alpha=0.7, label='Synthetic', bins=30, density=True)
                    axes[i].set_title(f'Distribution Comparison: {col}')
                    axes[i].legend()
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Density')
            
            # Remove empty subplots
            for i in range(len(numeric_cols), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Correlation heatmap comparison
        if len(numeric_cols) >= 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Real data correlation
            real_corr = self.real_data[numeric_cols].corr()
            sns.heatmap(real_corr, annot=True, cmap='coolwarm', center=0, ax=ax1)
            ax1.set_title('Real Data Correlation Matrix')
            
            # Synthetic data correlation
            synthetic_corr = self.synthetic_data[numeric_cols].corr()
            sns.heatmap(synthetic_corr, annot=True, cmap='coolwarm', center=0, ax=ax2)
            ax2.set_title('Synthetic Data Correlation Matrix')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'correlation_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. PCA visualization
        if len(numeric_cols) >= 2:
            try:
                # Prepare data for PCA
                scaler = StandardScaler()
                real_scaled = scaler.fit_transform(self.real_data[numeric_cols].fillna(0))
                synthetic_scaled = scaler.transform(self.synthetic_data[numeric_cols].fillna(0))
                
                # Apply PCA
                pca = PCA(n_components=2)
                real_pca = pca.fit_transform(real_scaled)
                synthetic_pca = pca.transform(synthetic_scaled)
                
                plt.figure(figsize=(12, 8))
                plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.6, label='Real', s=20)
                plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], alpha=0.6, label='Synthetic', s=20)
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.title('PCA Visualization: Real vs Synthetic Data')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(plots_dir / 'pca_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"PCA visualization failed: {e}")
        
        # 4. Categorical distribution comparison
        categorical_cols = self.real_data.select_dtypes(exclude=[np.number]).columns
        
        if len(categorical_cols) > 0:
            n_cats = min(4, len(categorical_cols))
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, col in enumerate(categorical_cols[:n_cats]):
                real_counts = self.real_data[col].value_counts()
                synthetic_counts = self.synthetic_data[col].value_counts()
                
                # Align categories
                all_categories = set(real_counts.index) | set(synthetic_counts.index)
                real_aligned = real_counts.reindex(all_categories, fill_value=0)
                synthetic_aligned = synthetic_counts.reindex(all_categories, fill_value=0)
                
                x_pos = np.arange(len(all_categories))
                width = 0.35
                
                axes[i].bar(x_pos - width/2, real_aligned.values, width, label='Real', alpha=0.7)
                axes[i].bar(x_pos + width/2, synthetic_aligned.values, width, label='Synthetic', alpha=0.7)
                axes[i].set_title(f'Category Distribution: {col}')
                axes[i].set_xlabel('Categories')
                axes[i].set_ylabel('Count')
                axes[i].set_xticks(x_pos)
                axes[i].set_xticklabels(all_categories, rotation=45, ha='right')
                axes[i].legend()
            
            # Remove empty subplots
            for i in range(n_cats, len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'categorical_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return str(plots_dir)
    
    def generate_quality_report_impl(self):
        """Generate comprehensive quality report"""
        print("\nGENERATING QUALITY REPORT")
        print("-" * 40)
        
        # Calculate metrics
        statistical_metrics = self.calculate_statistical_metrics()
        correlation_analysis = self.calculate_correlation_analysis()
        ml_efficacy = self.calculate_machine_learning_efficacy()
        
        # Generate visualizations
        plots_dir = self.generate_visualizations()
        
        # Compile overall quality scores
        overall_scores = []
        
        # Statistical similarity scores
        for col, metrics in statistical_metrics.items():
            overall_scores.append(metrics['distribution_similarity'])
        
        # Correlation preservation score
        if 'correlation_preservation_score' in correlation_analysis:
            overall_scores.append(correlation_analysis['correlation_preservation_score'])
        
        # ML efficacy score
        if 'performance_ratio' in ml_efficacy:
            overall_scores.append(ml_efficacy['performance_ratio'])
        
        overall_quality_score = np.mean(overall_scores) if overall_scores else 0
        
        # Create report
        report = {
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'overall_quality_score': overall_quality_score,
            'data_summary': {
                'original_shape': self.real_data.shape,
                'synthetic_shape': self.synthetic_data.shape,
                'columns': list(self.real_data.columns),
                'numeric_columns': len(self.real_data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(self.real_data.select_dtypes(exclude=[np.number]).columns)
            },
            'statistical_metrics': statistical_metrics,
            'correlation_analysis': correlation_analysis,
            'ml_efficacy': ml_efficacy,
            'model_parameters': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'generator_dim': self.generator_dim,
                'discriminator_dim': self.discriminator_dim
            },
            'visualizations_path': plots_dir
        }
        
        # Save report
        base_name = Path(self.data_path).stem
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(self.output_path) / f"{base_name}_quality_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n📊 QUALITY REPORT SUMMARY")
        print("=" * 50)
        print(f"Overall Quality Score: {overall_quality_score:.3f} (0-1 scale)")
        print(f"Report saved to: {report_path}")
        print(f"Visualizations saved to: {plots_dir}")
        
        print(f"\n📈 Key Metrics:")
        print(f"  • Statistical Similarity: {np.mean([m['distribution_similarity'] for m in statistical_metrics.values()]):.3f}")
        if 'correlation_preservation_score' in correlation_analysis:
            print(f"  • Correlation Preservation: {correlation_analysis['correlation_preservation_score']:.3f}")
        if 'performance_ratio' in ml_efficacy:
            print(f"  • ML Performance Ratio: {ml_efficacy['performance_ratio']:.3f}")
        
        self.quality_report = report
        return True
    
    def run(self):
        """Main program execution"""
        self.welcome_message()
        
        # Get parameters
        self.get_data_parameters()
        self.get_model_parameters()
        self.get_output_parameters()
        
        # Process data
        if not self.load_and_validate_data():
            return False
        
        # Train model
        if not self.train_model():
            return False
        
        # Generate synthetic data
        if not self.generate_synthetic_data():
            return False
        
        # Save data
        if not self.save_data():
            return False
        
        print("\n🎉 SUCCESS!")
        print("="*60)
        print("Synthetic data generation completed successfully!")
        print(f"Output file: {self.output_file}")
        
        if self.generate_quality_report:
            if not self.generate_quality_report_impl():
                print("⚠️  Quality report generation failed, but synthetic data was created successfully.")
        
        return True


# Main execution
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.run()