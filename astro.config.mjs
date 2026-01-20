import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import tailwind from '@astrojs/tailwind';

// https://astro.build/config
export default defineConfig({
  site: 'https://lmfast.github.io',
  base: '/slmhub',
  integrations: [
    starlight({
      title: 'SLM Hub',
      description: 'Developer-centric documentation for Small Language Models (SLMs)',
      favicon: '/favicon.svg',
      logo: {
        src: './src/assets/logo.svg',
        alt: 'SLM Hub',
        replacesTitle: false,
      },
      social: {
        github: 'https://github.com/lmfast/slmhub',
      },
      sidebar: [
        {
          label: 'Start Here',
          link: '/start-here/',
        },
        {
          label: 'Principles',
          link: '/principles/',
        },
        {
          label: 'Learn',
          autogenerate: {
            directory: 'learn',
          },
        },
        {
          label: 'Models',
          autogenerate: {
            directory: 'models',
            collapsed: true,
          },
        },
        {
          label: 'Deploy',
          autogenerate: {
            directory: 'deploy',
          },
        },
        {
          label: 'Tools',
          autogenerate: {
            directory: 'tools',
          },
        },
        {
          label: 'Community',
          autogenerate: {
            directory: 'community',
          },
        },
      ],
      customCss: [
        './src/styles/custom.css',
      ],
      // components: {
      //   PageFrame: './src/components/PageFrame.astro',
      // },
      head: [
        {
          tag: 'meta',
          attrs: {
            name: 'theme-color',
            content: '#6366f1',
          },
        },
        {
          tag: 'meta',
          attrs: {
            name: 'viewport',
            content: 'width=device-width, initial-scale=1.0, viewport-fit=cover',
          },
        },
        {
          tag: 'meta',
          attrs: {
            name: 'description',
            content: 'Developer-centric documentation for Small Language Models (SLMs): learn → choose a model → deploy → optimize.',
          },
        },
        {
          tag: 'meta',
          attrs: {
            property: 'og:title',
            content: 'SLM Hub - Developer Documentation for Small Language Models',
          },
        },
        {
          tag: 'meta',
          attrs: {
            property: 'og:description',
            content: 'Practical guides, minimal explanations, copy/paste code that runs, and decision frameworks for SLMs.',
          },
        },
        {
          tag: 'meta',
          attrs: {
            property: 'og:type',
            content: 'website',
          },
        },
        {
          tag: 'meta',
          attrs: {
            name: 'twitter:card',
            content: 'summary_large_image',
          },
        },
      ],
      defaultLocale: 'root',
      locales: {
        root: {
          label: 'English',
          lang: 'en',
        },
      },
      expressiveCode: {
        themes: ['github-dark', 'github-light'],
        defaultProps: {
          wrap: true,
          showLineNumbers: true,
        },
        useStrictTypescript: true,
        frame: {
          editor: false,
        },
      },
      pagefind: true,
      lastUpdated: true,
      editLink: {
        baseUrl: 'https://github.com/lmfast/slmhub/edit/main/',
      },
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 4,
      },
      pagination: true,
    }),
    tailwind({
      applyBaseStyles: false,
    }),
  ],
  output: 'static',
});
