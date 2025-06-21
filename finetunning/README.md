# Run

Run in terminal commands

```bash
uv venv --seed
uv run --with jupyter jupyter lab
```

Copy Jupyter server address in the output of the terminal
Ex. `http://127.0.0.1:8888/lab?token=7bd8bdcebb53d85b2d9d9740b38d2203d03e47fe13fd454e`

In VSCode Open Jupyter notebook, in the right corner select
`Select Kernel` > `Select Another Kernel` > `Existing Jupyter Server` > paste the URL from terminal `http://127.0.0.1:8888/lab?token=7bd8bdcebb53d85b2d9d9740b38d2203d03e47fe13fd454e` > Enter
