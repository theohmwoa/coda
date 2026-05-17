import type { Skill } from '../lib/types';

export function SkillCard({ skill, onRun }: { skill: Skill; onRun: (s: Skill) => void }) {
  return (
    <button className="skill-card" onClick={() => onRun(skill)} type="button">
      {skill.emoji ? <div className="skill-card__emoji">{skill.emoji}</div> : null}
      <div>
        <div className="skill-card__name">{skill.display_name}</div>
        <div className="skill-card__desc">{skill.description}</div>
      </div>
      <div className="skill-card__sig" title={skill.signature}>{skill.signature}</div>
    </button>
  );
}

export function SkillGrid({ skills, onRun }: { skills: Skill[]; onRun: (s: Skill) => void }) {
  if (skills.length === 0) {
    return (
      <div className="skill-grid__empty">
        No skills yet. Type a task below to begin.
      </div>
    );
  }
  return (
    <div className="skill-grid">
      {skills.map((s) => (
        <SkillCard key={s.name} skill={s} onRun={onRun} />
      ))}
    </div>
  );
}
