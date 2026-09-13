import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const [repositoryOwner = '', repositoryName = 'logicNN'] = (process.env.GITHUB_REPOSITORY ?? '').split('/');
const isGitHubPagesBuild = process.env.GITHUB_ACTIONS === 'true' && repositoryOwner !== '';
const isRootPagesRepository = repositoryName.toLowerCase() === `${repositoryOwner.toLowerCase()}.github.io`;
const siteUrl = isGitHubPagesBuild ? `https://${repositoryOwner}.github.io` : 'http://localhost';
const siteBaseUrl = isGitHubPagesBuild && !isRootPagesRepository ? `/${repositoryName}/` : '/';

const config: Config = {
  title: 'logicNN',
  tagline: '論理ゲートニューラルネットワークの学習・評価・回路出力',
  url: siteUrl,
  baseUrl: siteBaseUrl,
  organizationName: repositoryOwner || undefined,
  projectName: repositoryName,
  trailingSlash: false,
  onBrokenLinks: 'throw',
  future: {
    v4: true,
  },
  i18n: {
    defaultLocale: 'ja',
    locales: ['ja'],
  },
  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          // __init__.pyに対応する文書も表示し、非公開フォルダとテストは除外する。
          exclude: ['**/_*/**', '**/*.test.{js,jsx,ts,tsx}', '**/__tests__/**'],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],
  themes: ['@docusaurus/theme-mermaid'],
  markdown: {
    mermaid: true,
  },
  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'logicNN',
      items: [
        {to: '/', label: 'ホーム', position: 'left'},
        {type: 'docSidebar', sidebarId: 'startupSidebar', label: 'スタートアップ', position: 'left'},
        {type: 'docSidebar', sidebarId: 'userGuideSidebar', label: 'ユーザーガイド', position: 'left'},
        {type: 'docSidebar', sidebarId: 'codeReferenceSidebar', label: 'コードリファレンス', position: 'left'},
        {type: 'docSidebar', sidebarId: 'specificationsSidebar', label: '仕様書', position: 'left'},
      ],
    },
    footer: {
      style: 'dark',
      copyright: `Copyright © ${new Date().getFullYear()} logicNN`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
