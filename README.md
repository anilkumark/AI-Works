# 📚 Daily Git Workflow Guide for AI Projects

## 🎯 Purpose
This guide helps you save your daily AI/ML work to GitHub automatically. Your repository: **https://github.com/anilkumark/AI-Works**

---

## 🔄 Daily Workflow (Manual Process)

### **At the End of Each Day:**

```bash
# 1. Navigate to your project folder
cd ~/Documents/My_AI_Works

# 2. Activate your conda environment
conda activate py312_env

# 3. Check what files you created/modified today
git status

# 4. Add all your new work
git add .

# 5. Commit with today's date and description
git commit -m "Daily work [DATE]: [What you worked on]"

# 6. Push to GitHub
git push
```

### **Example Commands:**
```bash
git add .
git commit -m "Daily work Aug 14: Added GPT-2 probability tree visualization"
git push
```

---

## 📝 **Commit Message Templates**

Use these formats for clear commit messages:

```bash
# For new features
git commit -m "Added: BERT fine-tuning notebook"

# For fixes
git commit -m "Fixed: Tokenization display bug in GPT-2 visualizer"

# For experiments
git commit -m "Experiment: Comparing different sampling methods"

# For daily work
git commit -m "Daily work Aug 14: Text generation with different search strategies"

# For learning projects
git commit -m "Learning: Implemented attention mechanism visualization"
```

---

## 🚀 **Quick Daily Commands (Copy-Paste Ready)**

### **Option 1: Add Everything**
```bash
cd ~/Documents/My_AI_Works && conda activate py312_env && git add . && git commit -m "Daily work $(date +%b_%d): AI projects update" && git push
```

### **Option 2: Interactive (Recommended)**
```bash
cd ~/Documents/My_AI_Works
conda activate py312_env
git status                    # See what changed
git add .                     # Add all changes
git commit -m "Daily work Aug 14: [DESCRIBE YOUR WORK]"
git push
```

---

## 📂 **File Organization Tips**

### **Folder Structure:**
```
My_AI_Works/
├── Notebooks/
│   ├── GPT2_experiments.ipynb
│   └── BERT_finetuning.ipynb
├── Scripts/
│   ├── Greedy_Search.py
│   ├── Greedy_Search2.py
│   └── beam_search.py
├── Data/
│   └── datasets/
├── Models/
│   └── saved_models/
└── README.md
```

### **File Naming Convention:**
- `YYYY_MM_DD_project_name.py` (e.g., `2025_08_14_gpt2_visualization.py`)
- `topic_version.py` (e.g., `text_generation_v2.py`)
- `model_experiment.py` (e.g., `bert_sentiment_analysis.py`)

---

## ⚡ **Automation Scripts**

### **Create a Daily Commit Script:**

**File: `daily_commit.bat` (Windows)**
```batch
@echo off
cd /d "C:\Users\ANIL\Documents\My_AI_Works"
call conda activate py312_env
git add .
set /p message="Enter today's work description: "
git commit -m "Daily work %date:~-4,4%_%date:~-10,2%_%date:~-7,2%: %message%"
git push
echo.
echo ✅ Work saved to GitHub!
pause
```

**File: `daily_commit.sh` (Git Bash)**
```bash
#!/bin/bash
cd ~/Documents/My_AI_Works
conda activate py312_env
echo "Files changed today:"
git status --short
echo ""
read -p "Enter today's work description: " message
git add .
git commit -m "Daily work $(date +%Y_%m_%d): $message"
git push
echo "✅ Work saved to GitHub!"
```

### **How to Use Automation:**

1. **Create the script file** in your `My_AI_Works` folder
2. **Make it executable** (for Git Bash): `chmod +x daily_commit.sh`
3. **Run it daily**: 
   - Windows: Double-click `daily_commit.bat`
   - Git Bash: `./daily_commit.sh`

---

## 🔍 **Checking Your Work Online**

### **View Your Repository:**
- **Main page:** https://github.com/anilkumark/AI-Works
- **Commit history:** https://github.com/anilkumark/AI-Works/commits/main
- **Individual files:** Click on any `.py` file to view code

### **GitHub Features You Can Use:**
- **Issues:** Track bugs or ideas
- **Wiki:** Document your learning journey
- **Releases:** Tag major milestones
- **README:** Describe your projects

---

## 🆘 **Troubleshooting Common Issues**

### **Problem: "Nothing to commit"**
```bash
# Check if files are actually changed
git status
ls -la  # See all files

# Add specific files manually
git add filename.py
```

### **Problem: "Permission denied"**
```bash
# Re-authenticate with GitHub
git config --global credential.helper manager-core
git push  # Will prompt for login
```

### **Problem: "Merge conflicts"**
```bash
# If someone else modified the repo
git pull origin main
# Resolve conflicts, then
git add .
git commit -m "Resolved merge conflicts"
git push
```

### **Problem: "Large files"**
```bash
# For files > 100MB, use Git LFS
git lfs track "*.model"
git add .gitattributes
git add large_file.model
git commit -m "Added large model file"
git push
```

---

## 📅 **Weekly/Monthly Tasks**

### **Weekly Review:**
```bash
# See your week's work
git log --oneline --since="1 week ago"

# Create a weekly summary
git commit -m "Week summary: Completed GPT-2 analysis, started BERT experiments"
```

### **Monthly Backup:**
```bash
# Create a release/tag for monthly milestone
git tag -a v1.0 -m "August 2025: GPT-2 and text generation projects"
git push origin v1.0
```

---

## 🎯 **Best Practices**

### **✅ Do:**
- Commit daily (even small changes)
- Write descriptive commit messages
- Organize files in folders
- Add comments to your code
- Update README with project descriptions

### **❌ Don't:**
- Commit passwords or API keys
- Add large binary files without Git LFS
- Make very large commits (break them down)
- Use vague commit messages like "updated code"

---

## 🏆 **Goal Setting**

### **Daily Goal:**
- [ ] Work on AI/ML projects
- [ ] Commit changes to GitHub
- [ ] Write clear commit messages

### **Weekly Goal:**
- [ ] Complete at least one significant project
- [ ] Document learnings in README
- [ ] Review and organize code

### **Monthly Goal:**
- [ ] Create a major release
- [ ] Share interesting projects
- [ ] Learn new techniques

---

## 📞 **Quick Reference**

| Command | Purpose |
|---------|---------|
| `git status` | See what files changed |
| `git add .` | Stage all changes |
| `git add filename.py` | Stage specific file |
| `git commit -m "message"` | Save changes with message |
| `git push` | Upload to GitHub |
| `git log --oneline` | See recent commits |
| `git pull` | Download latest changes |

---

## 🎉 **You're All Set!**

Your daily workflow is now:
1. **Code/experiment** with AI projects
2. **Run the daily commands** (manual or script)
3. **Check GitHub** to see your progress
4. **Repeat tomorrow!**

**Repository:** https://github.com/anilkumark/AI-Works