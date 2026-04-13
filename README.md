<img src="./assets/images/splash.png" alt="COONTROL-UFSC" />

## **Introduction**

## **How to Run**

Create env with python env, need python 3.12 or later.

```sh
python -m venv env

```
Activate env

```sh 
source env/bin/activate
```

Install all packages in requirements
```sh
pip install -r requirements.txt
```

Change to rust folder and compile rust client tcp
```sh
cargo build --release
```

Run application
```sh
python main
```