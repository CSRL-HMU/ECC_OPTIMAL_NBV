function setLegendFontSize(fig, fontSize)
    legends = findall(fig, 'Type', 'Legend');
    set(legends, 'FontSize', fontSize);
end
