import json
import sys

file = json.load(open('rules_%s.json'%sys.argv[1]))
syntax = True
total = 0

triggers = dict()
for relation in file:
    with open('src/main/resources/grammars_%s/%s_unit.yml'%(sys.argv[1],relation), 'w') as f:
        count = 0
        for trigger in file[relation]:
            rules = [r for r in file[relation][trigger]]
            rl = relation.replace('_slash_', '/')
    
            for rule in rules:
                triggers['%s_0_%d'%(rl, count)] = trigger
                try:
                    if syntax:
                        f.write('''
- name: ${label}_${count}_%d
  label: ${label}
  priority: ${rulepriority}
  pattern: |
    trigger =  %s
    subject: ${subject_type} = %s
    object: ${object_type} = %s\n'''%(count, trigger, ' '.join(rule['subj']), ' '.join(rule['obj'])))
#                     elif syntax == False and (high_prec is None or (rl in high_prec and count in high_prec[rl].get("0", []))):
#                         f.write('''
# - name: ${label}_${count}_%d
#   label: ${label}
#   priority: ${rulepriority}
#   type: token
#   pattern: |
#     @object: ${object_type} (/.+/)*%s(/.+/)* @subject: ${subject_type}'''%(count, trigger))
                    total += 1
                except UnicodeEncodeError:
                    pass
                count += 1
with open("triggers.json","w") as f:
    f.write(json.dumps(triggers))
