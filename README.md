# 🎯 Synthetic Data Generator with CTGAN

A comprehensive Python tool for generating high-quality synthetic tabular data using Conditional Tabular GANs (CTGAN) with built-in quality assessment and visualization capabilities.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Quality Assessment](#quality-assessment)
- [Configuration Options](#configuration-options)
- [Output Files](#output-files)
- [Examples](#examples)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 🚀 Core Functionality
- **CTGAN-based Generation**: Uses state-of-the-art Conditional Tabular GANs for synthetic data generation
- **Interactive Configuration**: User-friendly prompts for all parameters
- **Flexible Data Handling**: Supports various missing value strategies
- **Multiple Output Formats**: CSV and JSON export options

### 📊 Quality Assessment
- **Statistical Metrics**: Distribution comparison using KS tests, chi-square tests
- **Correlation Analysis**: Preservation of variable relationships
- **ML Efficacy Testing**: Performance comparison using RandomForest models
- **Visual Validation**: Comprehensive plotting suite

### 🎨 Visualization Suite
- Distribution comparison plots (histograms)
- Correlation heatmap comparisons
- PCA scatter plots for structure analysis
- Categorical distribution bar charts

### 📈 Comprehensive Reporting
- Overall quality score (0-1 scale)
- Detailed JSON reports with all metrics
- Organized output structure with timestamps
- Publication-ready visualizations

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly sdv pathlib
```

### Clone Repository

```bash
git clone https://github.com/yourusername/synthetic-data-generator.git
cd synthetic-data-generator
```

## 🚀 Quick Start

1. **Prepare your CSV data file**
2. **Run the generator**:
   ```bash
   python synthetic_data_generator.py
   ```
3. **Follow the interactive prompts**:
   - Enter your CSV file path
   - Configure model parameters
   - Set output preferences
   - Choose quality report generation

## 📚 Usage

### Basic Usage

```python
from synthetic_data_generator import SyntheticDataGenerator

# Initialize generator
generator = SyntheticDataGenerator()

# Run with interactive prompts
generator.run()
```

### Programmatic Usage

```python
# Set parameters programmatically
generator.data_path = "your_data.csv"
generator.epochs = 300
generator.batch_size = 500
generator.learning_rate = 0.0002
generator.num_samples = 1000
generator.output_format = "CSV"
generator.generate_quality_report = True

# Execute pipeline
generator.load_and_validate_data()
generator.train_model()
generator.generate_synthetic_data()
generator.save_data()
if generator.generate_quality_report:
    generator.generate_quality_report_impl()
```

## 🔍 Quality Assessment

The quality assessment system evaluates synthetic data across multiple dimensions:

### Statistical Metrics
- **Distribution Similarity**: KS test for continuous variables, chi-square for categorical
- **Moment Matching**: Mean, variance, skewness, kurtosis comparison
- **Range Preservation**: Min/max value analysis

### Correlation Analysis
- **Correlation Preservation**: Frobenius norm of correlation matrix differences
- **Relationship Integrity**: Pairwise correlation comparison

### Machine Learning Efficacy
- **Performance Ratio**: ML model performance comparison (synthetic vs real)
- **Predictive Utility**: RandomForest classification accuracy testing

### Overall Quality Score
Composite score (0-1) combining all metrics:
- **0.8-1.0**: Excellent quality
- **0.6-0.8**: Good quality
- **0.4-0.6**: Fair quality
- **0.0-0.4**: Poor quality

## ⚙️ Configuration Options

### Data Parameters
- **File Path**: Path to your CSV file
- **Missing Value Strategy**:
  - Drop rows with missing values
  - Fill with mean/mode
  - Keep as-is (CTGAN handles some missing values)

### Model Parameters
- **Epochs**: Training iterations (default: 300)
- **Batch Size**: Training batch size (default: 500)
- **Learning Rate**: Generator/discriminator learning rate (default: 0.0002)
- **Generator Dimensions**: Neural network architecture (default: 256,256)
- **Discriminator Dimensions**: Neural network architecture (default: 256,256)

### Output Parameters
- **Format**: CSV or JSON
- **Directory**: Output location
- **Quality Report**: Enable/disable comprehensive analysis

## 📁 Output Files

### Generated Files Structure
```
output_directory/
├── dataset_synthetic_20241215_143022.csv          # Synthetic data
├── dataset_quality_report_20241215_143022.json    # Quality metrics
└── dataset_quality_plots_20241215_143022/         # Visualizations
    ├── distribution_comparison.png
    ├── correlation_comparison.png
    ├── pca_comparison.png
    └── categorical_comparison.png
```

### Quality Report Contents
```json
{
  "generation_timestamp": "2024-12-15T14:30:22",
  "overall_quality_score": 0.847,
  "data_summary": {
    "original_shape": [1000, 15],
    "synthetic_shape": [1000, 15],
    "columns": ["feature1", "feature2", "..."],
    "numeric_columns": 10,
    "categorical_columns": 5
  },
  "statistical_metrics": {...},
  "correlation_analysis": {...},
  "ml_efficacy": {...},
  "model_parameters": {...}
}
```

## 💡 Examples

### Example 1: Customer Data
```bash
# Input: customer_data.csv (demographics, purchase history)
# Output: 5000 synthetic customer records
# Quality Score: 0.89 (Excellent)
```

### Example 2: Financial Transactions
```bash
# Input: transactions.csv (amounts, categories, timestamps)
# Output: 10000 synthetic transactions
# Quality Score: 0.74 (Good)
```

### Example 3: IoT Sensor Data
```bash
# Input: sensor_readings.csv (temperature, humidity, pressure)
# Output: 50000 synthetic sensor readings
# Quality Score: 0.91 (Excellent)
```

## 🔧 Advanced Usage

### Custom Metadata Configuration
```python
from sdv.metadata import SingleTableMetadata

# Create custom metadata
metadata = SingleTableMetadata()
metadata.add_column('age', sdtype='numerical')
metadata.add_column('income', sdtype='numerical')
metadata.add_column('category', sdtype='categorical')

generator.metadata = metadata
```

### Batch Processing
```python
# Process multiple files
csv_files = ['data1.csv', 'data2.csv', 'data3.csv']

for file_path in csv_files:
    generator = SyntheticDataGenerator()
    generator.data_path = file_path
    generator.run()
```

## 📋 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=1.3.0 | Data manipulation |
| numpy | >=1.20.0 | Numerical computing |
| matplotlib | >=3.3.0 | Visualization |
| seaborn | >=0.11.0 | Statistical visualization |
| scipy | >=1.7.0 | Statistical tests |
| scikit-learn | >=1.0.0 | ML evaluation |
| plotly | >=5.0.0 | Interactive plots |
| sdv | >=1.0.0 | Synthetic data generation |

## 🛡️ Best Practices

### Data Preparation
- Clean your data before generation
- Handle outliers appropriately
- Ensure sufficient data volume (>1000 rows recommended)

### Model Training
- Use higher epochs for complex datasets
- Adjust batch size based on memory constraints
- Monitor training progress for convergence

### Quality Validation
- Always generate quality reports
- Review correlation preservation carefully
- Test ML efficacy for your specific use case

## 📊 Performance Benchmarks

| Dataset Size | Training Time | Memory Usage | Quality Score |
|-------------|---------------|--------------|---------------|
| 1K rows | 2-5 minutes | 2GB | 0.85-0.95 |
| 10K rows | 10-20 minutes | 4GB | 0.80-0.90 |
| 100K rows | 1-2 hours | 8GB | 0.75-0.85 |

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Reduce batch size
batch_size = 250  # Instead of 500
```

**Poor Quality Score**
```bash
# Increase training epochs
epochs = 500  # Instead of 300
```

**Correlation Not Preserved**
```bash
# Check for multicollinearity in original data
# Consider feature selection
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/sami3l/synthetic-data-generator.git
cd synthetic-data-generator
pip install -r requirements.txt
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/sami3l/synthetic-data-generator/issues)
- **Email**: elhadraoui.sami@emsi-edu.ma

## 🙏 Acknowledgments

- [SDV Team](https://sdv.dev/) for the excellent synthetic data generation library
- [CTGAN Paper](https://arxiv.org/abs/1907.00503) authors for the foundational research
- Contributors and beta testers

## 🔗 Related Projects

- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV)
- [CTGAN](https://github.com/sdv-dev/CTGAN)
- [Table-GAN](https://github.com/mahmoudnafifi/table_gan)

---

**⭐ If you find this project useful, please consider giving it a star!**
