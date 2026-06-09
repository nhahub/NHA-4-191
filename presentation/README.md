# Road-Sense Presentation Website

A static presentation website for the Road-Sense project documentation, reports, and visualizations.

## 🌐 Live Demo

Visit the live site at: [https://Abdallah4Z.github.io/Road-Sense/](https://Abdallah4Z.github.io/Road-Sense/)

## 📁 Structure

```
presentation/
├── index.html          # Home page with hero and results
├── docs.html           # Documentation viewer
├── reports.html        # Reports viewer
├── visuals.html        # Visualization gallery
├── styles.css          # Global styles
└── script.js           # JavaScript logic
```

## 🚀 Features

- **Multi-page architecture** with separate pages for docs, reports, and visuals
- **Dark/Light mode toggle** (night mode is default)
- **Responsive design** for mobile and desktop
- **Document viewer** with sidebar navigation and search
- **Image gallery** with filtering and full-screen viewer
- **Interactive charts** that adapt to theme

## 🛠️ Local Development

1. **Create symlinks** so the site can find the markdown and image files locally:
   ```bash
   cd presentation
   ln -s ../docs .
   ln -s ../reports .
   ln -s ../experiments .
   ```

2. **Run a local server**:
   ```bash
   python3 -m http.server 8000
   ```

3. Open `http://localhost:8000` in your browser.

## 📦 Deployment

The site is automatically deployed to GitHub Pages via GitHub Actions:

1. Push changes to `main` or `master` branch
2. GitHub Actions builds and deploys the site
3. Site is available at: `https://Abdallah4Z.github.io/Road-Sense/`

### Manual Deployment

You can also trigger deployment manually:
1. Go to **Actions** tab in your GitHub repository
2. Select **Deploy Presentation Website**
3. Click **Run workflow**

## 🎨 Customization

### Colors and Themes

Edit the CSS variables in `styles.css`:

```css
:root {
  /* Night mode (default) */
  --bg: #0f172a;
  --ink: #e2e8f0;
  /* ... */
}

body.light-mode {
  /* Light mode */
  --bg: #ffffff;
  --ink: #0f172a;
  /* ... */
}
```

### Adding Documents

Add new entries to the `docs` or `reports` arrays in `script.js`:

```javascript
const docs = [
  {
    title: "Document Title",
    description: "Brief description",
    path: "./docs/your-document.md",
    category: "category-name"
  }
];
```

### Adding Visualizations

Add new entries to the `visuals` array in `script.js`:

```javascript
const visuals = [
  {
    title: "Visualization Title",
    src: "./experiments/visualization/your-image.png",
    category: "dataset" // or "runs"
  }
];
```

## 📝 Notes

- Markdown documents are fetched from relative paths and rendered client-side
- Images should be placed in the `experiments/visualization/` directory
- The site uses no external dependencies except Google Fonts
- All JavaScript runs client-side with no build step required
