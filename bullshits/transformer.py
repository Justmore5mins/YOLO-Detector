from sys import argv

output:list[str] = []
i:int = 1

with open(argv[1],"r") as file:
    for item in file.readlines():
        output.append(f"N{i} {item}") if item != "\n" else None
        i += 1;

with open(argv[1],"w") as file:
    file.writelines(output)