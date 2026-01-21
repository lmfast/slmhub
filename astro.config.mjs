import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import tailwind from '@astrojs/tailwind';

// https://astro.build/config
export default defineConfig({
  site: 'https://lmfast.github.io',
  base: '/slmhub',
  image: {
    service: { entrypoint: 'astro/assets/services/noop' }
  },
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
      // Disable default index page to allow custom homepage
      disable404Route: false,
      sidebar: [
        {
          label: 'Home',
          link: '/',
        },
        {
          label: 'Course',
          items: [
            { label: 'Start Here', link: '/docs/start-here/' },
            {
              label: 'Learn',
              autogenerate: { directory: 'docs/learn' },
            },
            {
              label: 'Deploy',
              autogenerate: { directory: 'docs/deploy' },
            },
            {
              label: 'Models',
              autogenerate: { directory: 'docs/models', collapsed: true },
            },
            {
              label: 'Tools',
              autogenerate: { directory: 'docs/tools' },
            },
          ],
        },
        {
          label: 'Foundations',
          autogenerate: { directory: 'docs/learn/fundamentals' },
        },
        {
          label: 'Community',
          items: [
            { label: 'Overview', autogenerate: { directory: 'docs/community' } },
          ],
        },
        {
          label: 'About',
          link: '/docs/about/',
        },
        {
          label: 'Contributing',
          link: '/docs/community/contributing/',
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
            content: '#FF6B35',
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
      // UI Translations
      components: {
        SiteTitle: './src/components/SiteTitle.astro',
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
    }),
    tailwind({
      applyBaseStyles: false,
    }),
  ],
  output: 'static',
});
