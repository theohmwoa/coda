import { marked } from 'marked';
import { useMemo } from 'react';

export function AgentResponse({ markdown }: { markdown: string }) {
  const html = useMemo(() => {
    marked.setOptions({ gfm: true, breaks: false });
    return marked.parse(markdown) as string;
  }, [markdown]);
  return (
    <div className="agent-doc fade-in">
      <div className="agent-doc__kicker">coda · response</div>
      <div dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}
