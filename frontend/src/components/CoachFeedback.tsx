import { MessageCircle, ExternalLink } from 'lucide-react'
import type { Language } from '../api/client'

export default function CoachFeedback({ language, wandbUrl, cluster }: {
  language: Language
  wandbUrl?: string
  cluster: string
}) {
  return (
    <div className="bg-[#171717] rounded-2xl p-6 flex flex-col gap-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <MessageCircle size={15} className="text-indigo-400" />
          <span className="text-zinc-400 text-sm font-medium uppercase tracking-wider">AI Coach</span>
        </div>
        <span className="text-xs text-zinc-600 bg-zinc-800 px-2 py-1 rounded-lg font-mono">{cluster}</span>
      </div>

      {language.summary && (
        <p className="text-zinc-300 text-sm leading-relaxed">{language.summary}</p>
      )}

      {language.coach_feedback && (
        <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-xl p-4">
          <p className="text-indigo-200 text-sm leading-relaxed">{language.coach_feedback}</p>
        </div>
      )}

      {language.reasoning_trace_short && (
        <p className="text-zinc-600 text-xs leading-relaxed italic">{language.reasoning_trace_short}</p>
      )}

      {wandbUrl && (
        <a
          href={wandbUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-xs text-yellow-400 hover:text-yellow-300 transition-colors mt-1"
        >
          <span className="w-4 h-4 rounded bg-yellow-400/20 flex items-center justify-center text-[10px] font-bold text-yellow-400">W</span>
          View WandB Run
          <ExternalLink size={11} />
        </a>
      )}
    </div>
  )
}
