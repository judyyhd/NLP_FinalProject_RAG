# Setting Up GitHub Pages for Your NLP Project

## What I've Created

I've set up a GitHub Pages structure for your blog-style project report. Here's what's included:

### Files Created:

1. **`index.md`** - Main blog post with complete structure
   - Abstract, Introduction, Background
   - Methodology, Experiments, Results
   - Discussion, Conclusion, References
   - All sections with placeholders for your content

2. **`results.md`** - Detailed results page
   - Performance metrics tables
   - Visual analysis with all your plots
   - Statistical comparisons
   - Links to download full results

3. **`_config.yml`** - Jekyll configuration
   - Theme: minimal (clean, professional)
   - Site title and description
   - Navigation links

4. **`gh-pages` branch** - Separate branch for GitHub Pages

---

## Next Steps to Activate GitHub Pages

### 1. Push the gh-pages branch:
```bash
cd /home/hy1331/NLP_FinalProject_RAG
git add index.md results.md _config.yml SETUP_GITHUB_PAGES.md
git commit -m "Add GitHub Pages blog structure"
git push origin gh-pages
```

### 2. Enable GitHub Pages on GitHub:
1. Go to your repository: https://github.com/judyyhd/NLP_FinalProject_RAG
2. Click **Settings** > **Pages** (left sidebar)
3. Under "Source", select:
   - Branch: `gh-pages`
   - Folder: `/ (root)`
4. Click **Save**

### 3. Wait 1-2 minutes, then visit:
```
https://judyyhd.github.io/NLP_FinalProject_RAG/
```

---

## Customizing Your Blog

### Fill in Your Content

**`index.md`** - Main sections to complete:
- [ ] Abstract: Summarize your findings
- [ ] Introduction: Add motivation and research questions
- [ ] Background: Describe HotpotQA and RAG approaches
- [ ] Methodology: Detail your experimental setup
- [ ] Results: Add your actual numbers and findings
- [ ] Discussion: Analyze and interpret results
- [ ] Conclusion: Summarize key takeaways
- [ ] Team member names and contributions

**`results.md`** - Add:
- [ ] Actual performance numbers in tables
- [ ] Statistical significance tests
- [ ] Per-dataset analysis observations

### Add More Pages (Optional)

Create additional markdown files for specific topics:
```bash
# Example: detailed methodology
touch methodology.md

# Example: case studies
touch case-studies.md
```

Link them in your `_config.yml` navigation or reference in `index.md`.

---

## Changing the Theme

The default theme is `minimal`. To change it:

1. Edit `_config.yml`, replace the theme line:
```yaml
theme: jekyll-theme-cayman        # Clean, professional
theme: jekyll-theme-slate         # Dark theme
theme: jekyll-theme-architect     # Bold headers
theme: jekyll-theme-leap-day      # Colorful
```

2. See all themes: https://pages.github.com/themes/

---

## Advanced Customization

### Custom CSS
Create `assets/css/style.scss`:
```scss
---
---

@import "{{ site.theme }}";

/* Your custom styles */
.main-content {
  max-width: 900px;
}

h1 {
  color: #57068c;
}
```

### Add a Logo
In `_config.yml`:
```yaml
logo: /path/to/logo.png
```

### Add Google Analytics
In `_config.yml`:
```yaml
google_analytics: UA-XXXXXXXX-X
```

---

## Tips for a Great Blog Post

### 1. Use Visual Hierarchy
- Clear section headers
- Short paragraphs (3-5 sentences)
- Bullet points for lists
- Bold for emphasis

### 2. Embed Visualizations
Your plots are already in `evaluation/outputs/`. They'll display automatically with:
```markdown
![Description](evaluation/outputs/plot_name.png)
```

### 3. Add Interactive Elements
Consider adding:
- Collapsible sections for detailed info
- Tables with sortable columns
- Code snippets with syntax highlighting

### 4. Write for Different Audiences
- **Abstract:** High-level for everyone
- **Technical sections:** Details for researchers
- **Discussion:** Insights for practitioners

### 5. Tell a Story
Structure your blog like a narrative:
1. **Hook:** Why this matters (Introduction)
2. **Setup:** What you did (Methodology)
3. **Reveal:** What you found (Results)
4. **Meaning:** Why it matters (Discussion)
5. **Closure:** What's next (Conclusion)

---

## Troubleshooting

### Page not showing?
- Check that GitHub Pages is enabled in Settings
- Verify `gh-pages` branch exists
- Wait 2-3 minutes for deployment
- Check build status in Actions tab

### Images not loading?
- Ensure image paths are relative: `evaluation/outputs/plot.png`
- Images must be committed to the `gh-pages` branch
- Check image file names (case-sensitive on web)

### Markdown not rendering?
- Verify YAML front matter at top of each `.md` file:
  ```yaml
  ---
  layout: default
  title: Your Title
  ---
  ```

---

## Working with Two Branches

You'll have two branches:
- **`main`:** Your code, scripts, and documentation
- **`gh-pages`:** Your blog post and visualizations

### To update the blog:
```bash
# Switch to gh-pages
git checkout gh-pages

# Make changes to index.md, results.md, etc.
git add .
git commit -m "Update blog content"
git push origin gh-pages

# Switch back to main
git checkout main
```

### To copy new plots to blog:
```bash
# On main branch, generate new plots
# Switch to gh-pages
git checkout gh-pages

# Copy plots from main branch
git checkout main -- evaluation/outputs/
git add evaluation/outputs/
git commit -m "Update plots"
git push origin gh-pages
```

---

## Project Structure

```
NLP_FinalProject_RAG/
├── main branch (code repository)
│   ├── data_chunking/
│   ├── no_rag_vanilla_rag/
│   ├── evaluation/
│   ├── instructrag/
│   └── README.md
│
└── gh-pages branch (blog/website)
    ├── index.md              # Main blog post
    ├── results.md            # Detailed results page
    ├── _config.yml           # Jekyll configuration
    ├── evaluation/outputs/   # Plots and visualizations
    └── assets/              # (optional) CSS, images
```

---

## Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Jekyll Themes](https://pages.github.com/themes/)
- [Markdown Guide](https://www.markdownguide.org/)
- [Writing Academic Blog Posts](https://medium.com/google-developer-experts/how-to-write-a-good-technical-blog-post-6b534c1e9ed1)

---

Good luck with your blog post! Fill in the content, add your insights, and you'll have a professional project showcase ready for your course.
