const fs = require('fs');
const source = fs.readFileSync('Assignment3_Translation.py', 'utf8');
const sections = source.split(/^# %%(?: \[markdown\])?\r?\n/gm);
const markers = [...source.matchAll(/^# %%( \[markdown\])?\r?\n/gm)];
const cells = markers.map((marker, index) => {
  const body = sections[index + 1].replace(/^\n+|\n+$/g, '') + '\n';
  const markdown = Boolean(marker[1]);
  const content = markdown
    ? body.split('\n').map(line => line.replace(/^# ?/, '')).join('\n')
    : body;
  return {
    id: `cell-${String(index + 1).padStart(2, '0')}`,
    cell_type: markdown ? 'markdown' : 'code',
    metadata: {},
    ...(markdown ? {} : { execution_count: null, outputs: [] }),
    source: content.match(/.*(?:\n|$)/g).filter(Boolean),
  };
});
const notebook = {
  cells,
  metadata: {
    kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
    language_info: { name: 'python' },
    colab: { gpuType: 'T4' },
  },
  nbformat: 4,
  nbformat_minor: 5,
};
fs.writeFileSync('Assignment3_Translation.ipynb', JSON.stringify(notebook, null, 2) + '\n');
console.log(`Wrote ${cells.length} cells`);
