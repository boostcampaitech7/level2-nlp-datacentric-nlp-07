# Step 1: Install required modules
echo "Installing necessary modules..."
pip install jupyter pipreqs nbconvert

# Step 2: Read .gitignore and gather paths to ignore
echo "Reading .gitignore for paths to exclude..."
ignore_paths=()
if [ -f ".gitignore" ]; then
    while IFS= read -r line; do
        # Ignore comments and empty lines
        if [[ ! "$line" =~ ^# && -n "$line" ]]; then
            ignore_paths+=("$line")
        fi
    done < .gitignore
fi

# Function to check if a path should be ignored
should_ignore() {
    local path=$1
    for ignore in "${ignore_paths[@]}"; do
        if [[ "$path" == $ignore* ]]; then
            return 0
        fi
    done
    return 1
}

# Step 3: Convert .ipynb files to .py files and collect file names
echo "Converting .ipynb files to .py files..."
converted_files=()
for notebook in $(find . -type f -name "*.ipynb"); do
    # Skip ignored paths
    if should_ignore "$notebook"; then
        echo "Skipping $notebook (ignored)"
        continue
    fi
    jupyter nbconvert --to script "$notebook"
    # Append the converted .py file to the list
    converted_files+=("${notebook%.ipynb}.py")
done

# Step 4: Generate requirements.txt with pipreqs, ignoring specified directories
echo "Generating requirements.txt with pipreqs..."
pipreqs . --force --ignore "$(IFS=,; echo "${ignore_paths[*]}")"

# Step 5: Remove only the converted .py files
echo "Removing converted .py files..."
for file in "${converted_files[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
    fi
done

# Step 6: Check for dependency conflicts
echo "Checking for dependency conflicts..."
pip check

echo "Done! requirements.txt has been generated and dependencies have been checked for conflicts."