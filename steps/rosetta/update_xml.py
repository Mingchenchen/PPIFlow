import re
import argparse
from pathlib import Path


def parse_pdb(pdb_file):
    """
    从 PDB 文件中提取每条链的起始和结束残基编号。
    """
    chains = {}
    with open(pdb_file, "r") as file:
        for line in file:
            if line.startswith("ATOM"):
                chain_id = line[21]
                res_seq = int(line[22:26].strip())
                if chain_id not in chains:
                    chains[chain_id] = [res_seq, res_seq]
                else:
                    chains[chain_id][1] = res_seq
    return chains


def update_native_xml(native_xml_file, chains, output_path):
    """
    修改 XML 文件中的 resnums，并将更新后的内容保存为新文件。
    """
    with open(native_xml_file, "r") as file:
        content = file.read()

    resnums_pattern = r'resnums="[^"]+"'
    new_resnums = ",".join([f"{start}{chain}-{end}{chain}" for chain, (start, end) in chains.items()])
    new_content = re.sub(resnums_pattern, f'resnums="{new_resnums}"', content)

    with open(output_path, "w") as file:
        file.write(new_content)

    print(f"✅ 新 XML 文件保存到: {output_path}")
    print(f"👉 更新的 resnums: {new_resnums}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据 PDB 更新 XML 的 resnums 并保存新文件")
    parser.add_argument("--pdb_path", type=str, required=True, help="PDB 文件路径")
    args = parser.parse_args()

    chains = parse_pdb(args.pdb_path)

    script_dir = Path(__file__).resolve().parent
    xml_template = script_dir / "native.xml"
    output_xml = "update.xml"
    update_native_xml(xml_template, chains, output_xml)
