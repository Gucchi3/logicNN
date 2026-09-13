import {existsSync, readdirSync} from 'node:fs';
import path from 'node:path';
import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

type SidebarItems = Extract<SidebarsConfig[string], unknown[]>;

const projectRoot = path.resolve(__dirname, '..');
const docsRoot    = path.resolve(__dirname, 'docs/code-reference');
const sourceRoots = new Set(['main.py', 'model', 'tools', 'utils']);
const ignored     = new Set(['__pycache__', 'node_modules', 'build', 'dist']);

function sourceItems(directory: string): SidebarItems {
  /** 実際のPythonソース階層を読み、本文のないファイルは準備中として並べる。 */
  const items: SidebarItems = [];
  const entries = readdirSync(path.join(projectRoot, directory), {withFileTypes: true});
  entries.sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name, 'en'));
  for (const entry of entries) {
    if ((!directory && !sourceRoots.has(entry.name)) || entry.name.startsWith('.') || ignored.has(entry.name)) continue;
    const sourcePath = directory ? `${directory}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      const children = sourceItems(sourcePath);
      if (children.length) items.push({type: 'category', label: entry.name, collapsed: true, className: 'source-tree-directory', items: children});
    } else if (entry.isFile() && entry.name.endsWith('.py')) {
      const documentPath = sourcePath.slice(0, -3);
      if (existsSync(path.join(docsRoot, `${documentPath}.md`)) || existsSync(path.join(docsRoot, `${documentPath}.mdx`))) {
        items.push({type: 'doc', id: `code-reference/${documentPath}`, label: entry.name, className: 'source-tree-file'});
      } else {
        const label = entry.name.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
        items.push({type: 'html', defaultStyle: true, className: 'source-tree-pending', value: `<span>${label}</span><small>準備中</small>`});
      }
    }
  }
  return items;
}

const sourceTree = sourceItems('');
const mainIndex  = sourceTree.findIndex((item) => typeof item === 'object' && 'id' in item && item.id === 'code-reference/main');
const mainItem   = sourceTree.splice(mainIndex, 1)[0];

export default ['code-reference/index', mainItem, ...sourceTree] satisfies SidebarItems;
