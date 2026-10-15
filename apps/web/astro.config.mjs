// @ts-check
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import starlightLlmsTxt from 'starlight-llms-txt';
import sidebar from './src/generated/sidebar.json' with { type: 'json' };
import course from './src/generated/course.json' with { type: 'json' };

// Content is generated from the repo's READMEs by scripts/sync-content.mjs.
export default defineConfig({
	site: process.env.OPENLEARN_SITE_URL ?? 'https://openlearn.urmzd.com',
	trailingSlash: 'always',
	// The READMEs write LaTeX as $...$ and $$...$$, the way GitHub renders it.
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},
	integrations: [
		starlight({
			title: course.title,
			description: course.description,
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/urmzd/openlearn' }],
			customCss: ['katex/dist/katex.min.css', './src/styles/custom.css'],
			head: [
				{
					tag: 'script',
					attrs: { type: 'module' },
					content: `
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
const dark = document.documentElement.dataset.theme === 'dark';
mermaid.initialize({ startOnLoad: true, theme: dark ? 'dark' : 'neutral' });`,
				},
			],
			plugins: [starlightLlmsTxt()],
			sidebar,
		}),
	],
});
