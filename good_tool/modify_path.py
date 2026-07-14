import os

def modify_txt_paths(input_txt, output_txt, old_prefix, new_prefix):
    """
    修改文件路径
    Modify paths in a .txt file by replacing the old prefix with a new prefix
    and converting Linux-style paths to Windows-style paths.
    
    Args:
        input_txt (str): Path to the input .txt file
        output_txt (str): Path to the output .txt file
        old_prefix (str): Old path prefix to be replaced
        new_prefix (str): New path prefix to replace with
    """
    with open(input_txt, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        # Split the line into path and label
        path, label = line.strip().split(' ')
        # Replace the old prefix with the new one
        new_path = path.replace(old_prefix, new_prefix)
        # Convert linux-style/ forward slashes to windows-style\\ backslashes  ,在Python字符串中，反斜杠\是转义字符，所以用两个反斜杠\\表示一个实际的反斜杠
        new_path = new_path.replace('/', '\\')
        # Use os.path.normpath to ensure correct path separators
        new_path = os.path.normpath(new_path)
        # Combine the new path with the label
        modified_lines.append(f"{new_path} {label}\n")
    
    with open(output_txt, 'w') as f:
        f.writelines(modified_lines)


if __name__=='__main__':
    # Configuration
    '''
    Art Art_25
    Clipart Clipart_25
    Product Product_25
    RealWorld RealWorld_25
    
    '''
    input_txt = r'D:\LJ\workstation\Vscode\ISRA_changed\data\office_home\RealWorld_25_list.txt'  #  input .txt file path 需要修改的标签文件路径
    output_txt = r'D:\LJ\workstation\Vscode\ISRA_changed\data\office_home\RealWorld_25_list.txt'  #  output .txt file path 更新过的标签文件保存路径
    old_prefix = r'D:\LJ\workstation\Vscode\ISRA_changed\data\officehome'  #  old path prefix to be replaced 
    new_prefix = r'D:\LJ\workstation\Vscode\ISRA_changed\data\office_home' 

    # Run the modification
    modify_txt_paths(input_txt, output_txt, old_prefix, new_prefix)
    print(f"Paths modified successfully. Output saved to {output_txt}")