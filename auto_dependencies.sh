# Step 1: Install required modules
echo "Installing necessary modules..."
pip install jupyter pipreqs nbconvert

# Step 2: Backup existing requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    echo "Backing up existing requirements.txt to requirements_backup.txt..."
    mv requirements.txt requirements_backup.txt
fi

# Step 3: Convert all .ipynb files to .py files and track converted files
echo "Converting .ipynb files to .py files..."
converted_files=()
for notebook in *.ipynb; do
    jupyter nbconvert --to script "$notebook"
    # Extract the base name without extension
    base_name=$(basename "$notebook" .ipynb)
    # Append the converted .py file to the list
    converted_files+=("${base_name}.py")
done

# Step 4: Generate requirements.txt using pipreqs
echo "Generating requirements.txt with pipreqs..."
pipreqs . --force

# Step 5: Remove only the converted .py files
echo "Removing converted .py files..."
for file in "${converted_files[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
    fi
done

echo "Done! requirements.txt has been generated."
