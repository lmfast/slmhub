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
      // q Disable default index page to allow custom homepage
      disable404Route: false,
      sidebar: [
        {
          label: 'Home',
          link: '/',
        },
        {
          label: 'Learning',
          items: [
            { label: 'Start Here', link: '/docs/start-here/' },
            {
              label: 'Models',
              autogenerate: { directory: 'docs/learn/models' },
            },
            {
              label: 'Advanced Topics',
              autogenerate: { directory: 'docs/learn/advanced-topics' },
            },
            {
              label: 'Advanced Architectures',
              autogenerate: { directory: 'docs/learn/advanced-architectures' },
            },
            {
              label: 'Cutting-Edge',
              autogenerate: { directory: 'docs/learn/cutting-edge' },
            },
            {
              label: 'Mathematics',
              autogenerate: { directory: 'docs/learn/mathematics' },
              collapsed: true,
            },
          ],
        },
        {
          label: 'Foundations',
          link: '/docs/learn/foundations/',
        },
        {
          label: 'Hands-On',
          items: [
            {
              label: 'Tutorials',
              autogenerate: { directory: 'docs/learn/hands-on' },
            },
            {
              label: 'Projects',
              autogenerate: { directory: 'docs/learn/projects' },
            },
          ],
        },
        {
          label: 'Deployment',
          autogenerate: { directory: 'docs/deploy' },
        },
        {
          label: 'Tools & Resources',
          items: [
            { label: 'Interactive Tools', autogenerate: { directory: 'docs/tools' } },
            { label: 'Model Directory', link: '/docs/models/generated/directory/' },
            { label: 'Model Pages', autogenerate: { directory: 'docs/models', collapsed: true } },
          ],
        },
        {
          label: 'Community',
          link: '/docs/community/',
        },
        {
          label: 'Contributing',
          link: '/docs/community/contributing/',
        },
        {
          label: 'About',
          link: '/docs/about/',
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
        SocialIcons: './src/components/SocialIcons.astro',
        Footer: './src/components/Footer.astro',
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
      tableOfContents: false,
    }),
    tailwind({
      applyBaseStyles: false,
    }),
  ],
  output: 'static',
});
