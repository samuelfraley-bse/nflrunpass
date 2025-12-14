from setuptools import setup, find_packages

setup(
    name="nflplaycaller",
    version="0.1.0",
    description="NFL Play Call Predictor",
    author="Your Name",
    author_email="you@example.com",
    python_requires=">=3.10",
    package_dir={"": "src"},  # <-- tell setuptools to look in src/
    packages=find_packages(where="src"),  # <-- only find packages inside src/
    install_requires=[
        "pandas",
        "streamlit",
        "matplotlib",
        "joblib",
        "scikit-learn",
    ],
    include_package_data=True,
)
