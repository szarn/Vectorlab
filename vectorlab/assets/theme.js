function currentTheme() {
  const forced = document.documentElement.dataset.theme;
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  return THEMES[forced || (dark ? "dark" : "light")];
}

function paintFigure(gd, t) {
  const layout = {"font.color": t.ink2, "hoverlabel.bgcolor": t.surface, "hoverlabel.font.color": t.ink};
  Object.keys(gd.layout).filter(k => /^[xy]axis\d*$/.test(k)).forEach(k => {
    layout[k + ".gridcolor"] = t.grid;
    layout[k + ".linecolor"] = t.axis;
    layout[k + ".tickcolor"] = t.axis;
  });
  (gd.layout.annotations || []).forEach((a, i) => { layout["annotations[" + i + "].font.color"] = t.ink; });
  (gd.layout.shapes || []).forEach((s, i) => { layout["shapes[" + i + "].line.color"] = t.axis; });
  if (gd.layout.title && gd.layout.title.text) layout["title.font.color"] = t.ink;
  Plotly.Plots.resize(gd);
  Plotly.relayout(gd, layout);
  gd.data.forEach((trace, i) => {
    if (!trace.meta) return;
    const color = t[trace.meta];
    Plotly.restyle(gd, {"line.color": color, "marker.color": color, "marker.line.color": t.surface, "fillcolor": color + "1a"}, [i]);
  });
}

function paint() {
  const t = currentTheme();
  document.querySelectorAll(".js-plotly-plot").forEach(gd => paintFigure(gd, t));
}

window.addEventListener("load", paint);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paint);
