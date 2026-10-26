import argparse
import ast
import re


def parse_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    fold_pattern = re.compile(
        r"Fold\s*(\d+):\s*loss\s*=\s*([\d.]+)\s*\|\s*metrics\s*(\{.*?\})"
    )
    folds = []
    for match in fold_pattern.finditer(content):
        fold_idx = int(match.group(1))
        loss = float(match.group(2))
        metrics = ast.literal_eval(match.group(3))
        folds.append({
            "fold": fold_idx,
            "loss": loss,
            "mse": float(metrics.get("mse", 0.0)),
            "mae": float(metrics.get("mae", 0.0)),
            "r2": float(metrics.get("r2", 0.0))
        })

    summary_patterns = {
        "mse": r"MSE:\s*([\d.]+)\s*\(std:\s*([\d.]+)\)",
        "mae": r"MAE:\s*([\d.]+)\s*\(std:\s*([\d.]+)\)",
        "r2": r"R2:\s*([\d.]+)\s*\(std:\s*([\d.]+)\)"
    }
    summary = {}
    for key, pattern in summary_patterns.items():
        m = re.search(pattern, content)
        if m:
            summary[key] = (float(m.group(1)), float(m.group(2)))
        else:
            summary[key] = (0.0, 0.0)

    return folds, summary


def generate_latex_table(folds, summary, description, label):
    rows = []
    for row in folds:
        rows.append(
            f"        Fold {row['fold']} & {row['loss']:.4f} & {row['mse']:.4f} & {row['mae']:.4f} & {row['r2']:.4f} \\\\"
        )
    rows_text = "\n".join(rows)

    mse_mean, mse_std = summary["mse"]
    mae_mean, mae_std = summary["mae"]
    r2_mean, r2_std = summary["r2"]

    caption = f"Wyniki 5-krotnej walidacji krzyżowej dla {description}"

    latex_code = f"""\\begin{{table}}[H]
    \\centering
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\begin{{tabular}}{{l c c c c}}
        \\toprule
        \\textbf{{Fold}} & \\textbf{{Loss}} & \\textbf{{MSE}} & \\textbf{{MAE}} & \\textbf{{$R^2$}} \\\\
        \\midrule
{rows_text}
        \\bottomrule
    \\end{{tabular}}

    \\vspace{{0.4em}}
    {{\\small \\textit{{Wartości średnie:}} \\textbf{{MSE = {mse_mean:.4f} $\\pm$ {mse_std:.4f}}}, \\textbf{{MAE = {mae_mean:.4f} $\\pm$ {mae_std:.4f}}}, \\textbf{{$R^2$ = {r2_mean:.4f} $\\pm$ {r2_std:.4f}}}}}
\\end{{table}}"""
    return latex_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Ścieżka do pliku txt")
    parser.add_argument("desc",
                        help="Dokończenie opisu po 'Wyniki 5-krotnej walidacji krzyżowej dla '")
    parser.add_argument("--label", "-l", default="tab:cv_folds", help="Etykieta tabeli")
    args = parser.parse_args()

    folds, summary = parse_file(args.file)
    latex_output = generate_latex_table(folds, summary, description=args.desc,
                                        label=args.label)
    print(latex_output)


if __name__ == "__main__":
    main()