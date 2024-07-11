# ChronIO: A toolbox for analysis of physiological time series data

### Overview
ChronIO is a powerful Python toolbox designed to simplify the analysis of time series data across various scientific disciplines. While it excels in neuroscience and behavioral studies, ChronIO's flexibility makes it valuable for a wide range of applications, including:

- Neuroscience: Analyze neural activity patterns and correlate them with behavior
- Physiology: Process and interpret physiological signals from wearables and sensors \(i.e. ECG, EMG, or respiratory data)
- Sports Science: Track athletic performance metrics and identify patterns

ChronIO empowers researchers and data scientists to efficiently manage, process, and analyze complex time series datasets, regardless of their programming expertise.

### Key Advantages 
ChronIO addresses several common pain points in time series analysis:

- **Unified Data Structure**: Seamlessly handle diverse data types within a consistent framework.
- **Intuitive Interface**: Simplify complex operations with an easy-to-use API, accessible to beginners and experts alike.
- **Flexible Processing**: Apply various transformations and analyses to your data with built-in functions.
- **Efficient Data Management**: Organize and access your data effortlessly, reducing the cognitive load of data handling.
- **Reproducibility**: Promote reproducible research with standardized data formats and processing pipelines.

### FAIR Data Principles
ChronIO is designed with FAIR (Findable, Accessible, Interoperable, Reusable) data principles in mind:

- **Findable**: Standardized naming conventions and metadata make it easy to locate specific datasets.
- **Accessible**: Open-source nature ensures that the tools for accessing data are freely available.
- **Interoperable**: Common data structures facilitate integration with other tools and datasets.
- **Reusable**: Detailed metadata and reproducible processing steps enhance data reusability.


### Quick Start Example
Here's a simple example demonstrating how to use ChronIO to read, process, and export data:
```
from chronio import BehavioralTimeSeries, Convention
import chronio.analyses as analyses

# Load data
data = BehavioralTimeSeries(fpath="subject0_behavior.csv", time_col="Time")

# Downsample and normalize
data.downsample_by_time(interval=0.1, method='mean', inplace=True)
data.normalize(columns=['Speed', 'Orientation'], inplace=True)

# Get events and export
events = data.get_events(columns=['ButtonPress'])
convention = Convention(directory="output", suffix="csv", metadata_fields=["subject_id", "session"])
events['LeverPress'].export(convention)
```

### Installation
To install ChronIO via Anaconda, follow these steps:

1. Clone the repository:

```
git clone https://github.com/alimoges42/chronio.git
cd chronio
```

2. Create and activate the conda environment:

```
conda env create -f environment.yml
conda activate chronio
```

3. Install ChronIO locally:

```
pip install .    # (or "pip install -e." for development mode)
```

This will install ChronIO and all its dependencies, allowing you to easily update the package as new changes are made.


### Contributing
We welcome contributions! If there is a functionality you are interested in, please open an issue or send a pull request to get started.

### License
ChronIO is released under the GPL-3 License. See the LICENSE file for details.

### Developer Information
Aaron Limoges (Github: [@alimoges42](https://github.com/alimoges42))
